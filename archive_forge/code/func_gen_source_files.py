import argparse
import functools
import json
import os
import pathlib
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from typing import (
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.gen_functionalization_type import (
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
from torchgen.yaml_utils import YamlDumper, YamlLoader
def gen_source_files(*, native_functions: Sequence[NativeFunction], grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], structured_native_functions: Sequence[NativeFunctionsGroup], view_groups: Sequence[NativeFunctionsViewGroup], selector: SelectiveBuilder, static_dispatch_idx: List[BackendIndex], backend_indices: Dict[DispatchKey, BackendIndex], core_fm: FileManager, cpu_fm: FileManager, cpu_vec_fm: FileManager, cuda_fm: FileManager, dispatch_keys: Sequence[DispatchKey], functions_keys: Set[DispatchKey], rocm: bool, force_schema_registration: bool, per_operator_headers: bool, skip_dispatcher_op_registration: bool) -> None:
    extra_cuda_headers = '#include <c10/cuda/CUDAGuard.h>\n#include <ATen/cuda/ATenCUDAGeneral.h>\n#include <ATen/cuda/CUDADevice.h>\n#include <ATen/cuda/CUDAContext.h>'
    if rocm:
        extra_cuda_headers = '#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>\n#include <ATen/hip/ATenHIPGeneral.h>\n#include <ATen/hip/HIPDevice.h>\n#include <ATen/hip/HIPContext.h>'
    for dispatch_key in dispatch_keys:
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm
        if per_operator_headers:

            def operator_headers() -> List[str]:
                headers = []
                for g in grouped_native_functions:
                    is_registered = False
                    if backend_index.has_kernel(g):
                        is_registered = True
                    elif isinstance(g, NativeFunctionsGroup) and any((backend_index.has_kernel(fn) for fn in g.functions())):
                        is_registered = True
                    elif g.structured and dispatch_key in (DispatchKey.Meta, DispatchKey.CompositeExplicitAutogradNonFunctional):
                        is_registered = True
                    if not is_registered:
                        continue
                    headers.append(f'#include <ATen/ops/{g.root_name}_native.h>')
                    if dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
                        headers.append(f'#include <ATen/ops/{g.root_name}.h>')
                    if dispatch_key in functions_keys:
                        headers.append(f'#include <ATen/ops/{g.root_name}_{dispatch_namespace}_dispatch.h>')
                return sorted(set(headers))
        else:

            def operator_headers() -> List[str]:
                headers = ['#include <ATen/NativeFunctions.h>']
                if dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
                    headers.append('#include <ATen/Functions.h>')
                if dispatch_key in functions_keys:
                    headers.append(f'#include <ATen/{dispatch_key!s}Functions.h>')
                return headers
        backend_index = backend_indices[dispatch_key]
        ns_grouped_native_functions = defaultdict(list)
        for grouped_native_function in grouped_native_functions:
            namespace = grouped_native_function.namespace if isinstance(grouped_native_function, NativeFunction) else grouped_native_function.functional.namespace
            ns_grouped_native_functions[namespace].append(grouped_native_function)
        dispatch_namespace = str(dispatch_key).lower()
        gen_dispatch_helpers: bool = dispatch_key != DispatchKey.CompositeImplicitAutogradNestedTensor
        dispatch_definitions = get_native_function_definitions(fm=fm, grouped_native_functions=grouped_native_functions, dispatch_key=dispatch_key, backend_idx=backend_index, selector=selector, rocm=rocm, symint=True, skip_dispatcher_op_registration=skip_dispatcher_op_registration, gen_dispatch_helpers=gen_dispatch_helpers)
        fm.write_with_template(f'Register{dispatch_key}.cpp', 'RegisterDispatchKey.cpp', lambda: {'extra_cuda_headers': extra_cuda_headers if is_cuda_dispatch_key(dispatch_key) else '', 'external_backend_headers': '', 'dispatch_headers': dest.gen_registration_headers(backend_index, per_operator_headers, rocm), 'ops_headers': operator_headers(), 'dispatch_helpers': '', 'dispatch_definitions': dispatch_definitions})
        for g in structured_native_functions:
            if not g.out.ufunc_inner_loop or not is_ufunc_dispatch_key(dispatch_key):
                continue
            name = g.functional.func.name.name
            if dispatch_key is DispatchKey.CPU:
                assert fm is cpu_fm
                fm.write_with_template(f'UfuncCPU_{name}.cpp', 'UfuncCPU.cpp', lambda: {'meta_declaration': compute_meta_function_declaration(g), 'native_declaration': dest.compute_native_function_declaration(g, backend_indices[dispatch_key]), 'native_definitions': dest.compute_ufunc_cpu(g)})
                cpu_vec_fm.write_with_template(f'UfuncCPUKernel_{name}.cpp', 'UfuncCPUKernel.cpp', lambda: {'name': name, 'native_definitions': dest.compute_ufunc_cpu_kernel(g)})
            elif dispatch_key is DispatchKey.CUDA:
                cuda_headers = '#include <ATen/native/cuda/Loops.cuh>'
                if rocm:
                    cuda_headers = '#include <ATen/native/hip/Loops.cuh>'
                fm.write_with_template(f'UfuncCUDA_{name}.cu', 'UfuncCUDA.cu', lambda: {'name': name, 'cuda_headers': cuda_headers, 'meta_declaration': compute_meta_function_declaration(g), 'native_declaration': dest.compute_native_function_declaration(g, backend_indices[dispatch_key]), 'native_definitions': dest.compute_ufunc_cuda(g)})
            else:
                raise AssertionError(f'unrecognized {dispatch_key} for ufunc')
        del fm

    def gen_backend_select() -> Dict[str, List[str]]:
        relevant_fns = [fn for fn in native_functions if needs_backend_select(fn, selector)]
        return {'ops_headers': [f'#include <ATen/ops/{fn.root_name}_ops.h>' for fn in relevant_fns], 'backend_select_method_definitions': list(mapMaybe(ComputeBackendSelect(Target.DEFINITION, selector), relevant_fns)), 'backend_select_function_registrations': list(mapMaybe(ComputeBackendSelect(Target.REGISTRATION, selector), relevant_fns))}
    cpu_fm.write('RegisterBackendSelect.cpp', gen_backend_select)
    schema_selector = selector
    if force_schema_registration:
        schema_selector = SelectiveBuilder.get_nop_selector()
    aten_schema_registrations, schema_registrations = get_native_function_schema_registrations(native_functions=native_functions, schema_selector=schema_selector)
    cpu_fm.write('RegisterSchema.cpp', lambda: {'aten_schema_registrations': [] if skip_dispatcher_op_registration else aten_schema_registrations, 'schema_registrations': [] if skip_dispatcher_op_registration else schema_registrations})

    def key_func(fn: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]) -> str:
        return fn.root_name
    cpu_fm.write_sharded('Operators.cpp', native_functions, key_fn=key_func, env_callable=lambda fn: {'operator_headers': [f'#include <ATen/ops/{fn.root_name}.h>'], 'definitions': [ComputeOperators(Target.DEFINITION, static_dispatch_backend_indices=static_dispatch_idx)(fn)]}, base_env={'static_dispatch_extra_headers': static_dispatch_extra_headers(static_dispatch_idx)}, num_shards=5, sharded_keys={'operator_headers', 'definitions', 'static_dispatch_extra_headers'})
    cpu_fm.write('Functions.cpp', lambda: {})
    core_fm.write('TensorMethods.cpp', lambda: {})
    core_fm.write('ATenOpList.cpp', lambda: {'aten_ops': list(mapMaybe(compute_aten_op, native_functions))})

    def functionalization_env_callable(g: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]) -> Dict[str, List[str]]:

        def gen_op_headers(g: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]) -> List[str]:
            if isinstance(g, NativeFunctionsViewGroup):
                headers = [f'#include <ATen/ops/{g.view.root_name}_native.h>', f'#include <ATen/ops/{g.view.root_name}_ops.h>']
                if g.view_copy is not None:
                    headers += [f'#include <ATen/ops/{g.view_copy.root_name}_native.h>', f'#include <ATen/ops/{g.view_copy.root_name}_ops.h>']
                return headers
            elif isinstance(g, NativeFunctionsGroup):
                headers = [f'#include <ATen/ops/{g.functional.root_name}_native.h>', f'#include <ATen/ops/{g.functional.root_name}_ops.h>', f'#include <ATen/ops/{g.out.root_name}_native.h>', f'#include <ATen/ops/{g.out.root_name}_ops.h>']
                if g.inplace is not None:
                    headers += [f'#include <ATen/ops/{g.inplace.root_name}_native.h>', f'#include <ATen/ops/{g.inplace.root_name}_ops.h>']
                if g.mutable is not None:
                    headers += [f'#include <ATen/ops/{g.mutable.root_name}_native.h>', f'#include <ATen/ops/{g.mutable.root_name}_ops.h>']
                return headers
            else:
                return [f'#include <ATen/ops/{g.root_name}_native.h>', f'#include <ATen/ops/{g.root_name}_ops.h>']
        return {'ops_headers': gen_op_headers(g), 'func_definitions': gen_functionalization_definition(selector, g), 'func_registrations': gen_functionalization_registration(selector, g, backend_indices[DispatchKey.CompositeImplicitAutograd])}
    all_groups: List[Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]] = list(structured_native_functions) + list(view_groups)
    structured_map: Dict[OperatorName, NativeFunction] = {f.func.name: f for f in concatMap(lambda g: list(g.functions()), structured_native_functions)}
    view_map: Dict[OperatorName, NativeFunction] = {f.func.name: f for f in concatMap(lambda g: list(g.functions()), view_groups)}
    for f in native_functions:
        if f.func.name not in structured_map and f.func.name not in view_map:
            all_groups.append(f)
    cpu_fm.write_sharded('RegisterFunctionalization.cpp', all_groups, key_fn=key_func, env_callable=functionalization_env_callable, num_shards=4, sharded_keys={'ops_headers', 'func_definitions', 'func_registrations', 'func_add_back_views_definitions', 'func_add_back_views_registrations'})
    cpu_fm.write('FunctionalInverses.h', lambda: {'view_inverse_declarations': list(mapMaybe(lambda g: gen_functionalization_view_inverse_declaration(selector, g), view_groups))})
    cpu_fm.write('CompositeViewCopyKernels.cpp', lambda: {'ops_headers': ['\n'.join((f'#include <ATen/ops/{f.root_name}_ops.h>\n#include <ATen/ops/{f.root_name}_native.h>' for f in ([g.view] if g.view_copy is None else [g.view, g.view_copy]))) for g in view_groups] + ['\n'.join((f'#include <ATen/ops/{f.root_name}_ops.h>' for f in [g.inplace, g.mutable, g.functional] if f is not None and 'generated' not in f.tags)) for g in structured_native_functions], 'CompositeViewCopyKernel_Definitions': list(mapMaybe(GenCompositeViewCopyKernel(backend_indices[DispatchKey.CompositeExplicitAutogradNonFunctional]), view_groups)), 'GeneratedCompositeFunctional_Definitions': list(mapMaybe(gen_composite_functional_kernel, structured_native_functions)), 'GeneratedCompositeOut_Definitions': list(mapMaybe(gen_composite_out_kernel, structured_native_functions))})