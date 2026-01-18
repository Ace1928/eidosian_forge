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
def gen_headers(*, native_functions: Sequence[NativeFunction], valid_tags: Set[str], grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], structured_native_functions: Sequence[NativeFunctionsGroup], static_dispatch_idx: List[BackendIndex], selector: SelectiveBuilder, backend_indices: Dict[DispatchKey, BackendIndex], core_fm: FileManager, cpu_fm: FileManager, cuda_fm: FileManager, ops_fm: FileManager, dispatch_keys: Sequence[DispatchKey], functions_keys: Set[DispatchKey], rocm: bool, per_operator_headers: bool) -> None:
    if per_operator_headers:
        gen_per_operator_headers(native_functions=native_functions, grouped_native_functions=grouped_native_functions, static_dispatch_idx=static_dispatch_idx, selector=selector, backend_indices=backend_indices, cpu_fm=cpu_fm, cuda_fm=cuda_fm, ops_fm=ops_fm, dispatch_keys=dispatch_keys, functions_keys=functions_keys, rocm=rocm)
    else:
        gen_aggregated_headers(native_functions=native_functions, grouped_native_functions=grouped_native_functions, structured_native_functions=structured_native_functions, static_dispatch_idx=static_dispatch_idx, selector=selector, backend_indices=backend_indices, cpu_fm=cpu_fm, cuda_fm=cuda_fm, dispatch_keys=dispatch_keys, functions_keys=functions_keys, rocm=rocm)
    core_fm.write('TensorBody.h', lambda: {'tensor_method_declarations': list(mapMaybe(ComputeTensorMethod(target=Target.DECLARATION, static_dispatch_backend_indices=static_dispatch_idx), native_functions)), 'tensor_method_definitions': list(mapMaybe(ComputeTensorMethod(target=Target.DEFINITION, static_dispatch_backend_indices=static_dispatch_idx), native_functions))})
    cpu_fm.write('RedispatchFunctions.h', lambda: {'function_redispatch_definitions': list(mapMaybe(ComputeRedispatchFunction(), native_functions))})
    cpu_fm.write('RegistrationDeclarations.h', lambda: {'registration_declarations': [compute_registration_declarations(f, backend_indices) for f in native_functions]})
    cpu_fm.write('VmapGeneratedPlumbing.h', lambda: gen_all_vmap_plumbing(native_functions))

    def gen_aten_interned_strings() -> Dict[str, str]:
        attrs = set()
        names = set()
        for func in native_functions:
            names.add(str(func.func.name.name))
            names.add(func.func.name.name.base)
            for arg in func.func.schema_order_arguments():
                attrs.add(arg.name)
        names -= {'and', 'and_eq', 'bitand', 'bitor', 'compl', 'not', 'not_eq', 'or', 'or_eq', 'xor', 'xor_eq'}
        return {'aten_symbols': ' \\\n'.join([f'_(aten, {name})' for name in sorted(names)]), 'attr_symbols': ' \\\n'.join([f'_(attr, {name})' for name in sorted(attrs)])}
    core_fm.write('aten_interned_strings.h', gen_aten_interned_strings)

    def gen_tags_enum() -> Dict[str, str]:
        return {'enum_of_valid_tags': ',\n'.join(sorted(valid_tags))}
    core_fm.write('enum_tag.h', gen_tags_enum)