import argparse
import os
import pathlib
import re
from collections import Counter, defaultdict, namedtuple
from typing import Dict, List, Optional, Sequence, Set, Union
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.dest as dest
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import native_function_manager
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, context, FileManager, NamespaceHelper, Target
from torchgen.yaml_utils import YamlLoader
def parse_backend_yaml(backend_yaml_path: str, grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], backend_indices: Dict[DispatchKey, BackendIndex]) -> ParsedExternalYaml:
    native_functions_map: Dict[OperatorName, NativeFunction] = {f.func.name: f for f in concatMap(lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()), grouped_native_functions)}
    with open(backend_yaml_path) as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    assert isinstance(yaml_values, dict)
    valid_keys = ['backend', 'class_name', 'cpp_namespace', 'extra_headers', 'supported', 'autograd', 'full_codegen', 'non_native', 'ir_gen', 'symint']
    backend = yaml_values.pop('backend', None)
    assert backend is not None, 'You must provide a value for "backend"'
    class_name = yaml_values.pop('class_name', None)
    cpp_namespace = yaml_values.pop('cpp_namespace', None)
    assert cpp_namespace is not None, 'You must provide a value for "cpp_namespace"'
    use_out_as_primary = yaml_values.pop('use_out_as_primary', False)
    assert isinstance(use_out_as_primary, bool), f'You must provide either True or False for use_out_as_primary. Provided: {use_out_as_primary}'
    use_device_guard = yaml_values.pop('device_guard', False)
    assert isinstance(use_device_guard, bool), f'You must provide either True or False for device_guard. Provided: {use_device_guard}'
    supported = yaml_values.pop('supported', [])
    if supported is None:
        supported = []
    assert isinstance(supported, list), f'expected "supported" to be a list, but got: {supported} (of type {type(supported)})'
    symint = yaml_values.pop('symint', [])
    if symint is None:
        symint = []
    assert isinstance(symint, list), f'expected "symint" to be a list, but got: {supported} (of type {type(supported)})'
    symint_set = set(symint)
    supported_autograd = yaml_values.pop('autograd', [])
    assert isinstance(supported_autograd, list), f'expected "autograd" to be a list, but got: {supported_autograd}'
    full_codegen = yaml_values.pop('full_codegen', [])
    supported.extend(full_codegen)
    non_native = yaml_values.pop('non_native', {})
    _ = yaml_values.pop('ir_gen', {})
    assert len(yaml_values.keys()) == 0, f'{backend_yaml_path} contains unexpected keys: {', '.join(yaml_values.keys())}. Only the following keys are supported: {', '.join(valid_keys)}'

    def create_backend_index(backend_ops: List[str], symint_ops: Set[str], dispatch_key: DispatchKey, *, use_out_as_primary: bool, use_device_guard: bool) -> BackendIndex:
        metadata: Dict[OperatorName, BackendMetadata] = {}
        for op in backend_ops:
            op_name = OperatorName.parse(op)
            assert op_name in native_functions_map, f'Found an invalid operator name: {op_name}'
            kernel_name = dispatcher.name(native_functions_map[op_name].func)
            if op in symint_ops:
                kernel_name += '_symint'
            m = BackendMetadata(kernel=kernel_name, structured=False, cpp_namespace=cpp_namespace)
            metadata[op_name] = m
        return BackendIndex(dispatch_key=dispatch_key, use_out_as_primary=use_out_as_primary, external=True, device_guard=use_device_guard, index=metadata)
    backend_key: Optional[DispatchKey] = None
    if len(supported) > 0:
        with context(lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'):
            backend_key = DispatchKey.parse(backend)
        backend_idx = create_backend_index(supported, symint_set, backend_key, use_out_as_primary=use_out_as_primary, use_device_guard=use_device_guard)
        assert backend_key not in backend_indices
        backend_indices[backend_key] = backend_idx
    autograd_key: Optional[DispatchKey] = None
    if len(supported_autograd) > 0:
        with context(lambda: f'The "autograd" key was specified, which indicates that you would like to override the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'):
            autograd_key = DispatchKey.parse(f'Autograd{backend}')
        autograd_idx = create_backend_index(supported_autograd, symint_set, autograd_key, use_out_as_primary=use_out_as_primary, use_device_guard=use_device_guard)
        assert autograd_key not in backend_indices
        backend_indices[autograd_key] = autograd_idx
    for g in grouped_native_functions:
        if isinstance(g, NativeFunction):
            forward_kernels = [] if backend_key is None else [m for m in [backend_indices[backend_key].get_kernel(g)] if m is not None]
            backward_kernels = [] if autograd_key is None else [m for m in [backend_indices[autograd_key].get_kernel(g)] if m is not None]
        else:
            forward_kernels = [] if backend_key is None else [m for m in [backend_indices[backend_key].get_kernel(f) for f in g.functions()] if m is not None]
            backward_kernels = [] if autograd_key is None else [m for m in [backend_indices[autograd_key].get_kernel(f) for f in g.functions()] if m is not None]
        forward_kernels = [f for f in forward_kernels if f is not None]
        backward_kernels = [f for f in backward_kernels if f is not None]
        assert len(forward_kernels) == 0 or len(backward_kernels) == 0, f"""Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! {forward_kernels[0].kernel} is listed under "supported", but {backward_kernels[0].kernel} is listed under "autograd"."""
    return ParsedExternalYaml(backend_key, autograd_key, class_name, cpp_namespace, backend_indices)