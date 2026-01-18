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
def error_on_missing_kernels(native_functions: Sequence[NativeFunction], backend_indices: Dict[DispatchKey, BackendIndex], backend_key: DispatchKey, autograd_key: Optional[DispatchKey], class_name: str, kernel_defn_file_path: str, full_codegen: Optional[List[OperatorName]]=None) -> None:
    try:
        with open(kernel_defn_file_path) as f:
            backend_defns = f.read()
    except OSError as e:
        raise AssertionError(f'Unable to read from the specified impl_path file: {kernel_defn_file_path}') from e
    if full_codegen is None:
        full_codegen = []
    indices = [backend_indices[backend_key].index] + ([] if autograd_key is None else [backend_indices[autograd_key].index])
    expected_backend_op_names: Dict[OperatorName, str] = dict(list(concatMap(lambda index: [(op_name, metadata.kernel) for op_name, metadata in index.items()], indices)))
    expected_backend_native_funcs: List[NativeFunction] = [f for f in native_functions if f.func.name in expected_backend_op_names.keys() and f.func.name not in full_codegen]
    expected_backend_kernel_name_counts: Dict[str, List[NativeFunction]] = defaultdict(list)
    for native_f in expected_backend_native_funcs:
        expected_backend_kernel_name_counts[expected_backend_op_names[native_f.func.name]].append(native_f)
    kernel_defn_regex = f'(.*){class_name}::\\s*([\\w\\d]*)\\('
    actual_backend_kernel_name_counts = Counter([y for x, y in re.findall(kernel_defn_regex, backend_defns) if not x.endswith(':')])
    missing_kernels_err_msg = ''
    for expected_name, funcs in expected_backend_kernel_name_counts.items():
        expected_overload_count = len(funcs)
        actual_overload_count = actual_backend_kernel_name_counts[expected_name]
        if expected_overload_count != actual_overload_count:

            def create_decl(f: NativeFunction) -> str:
                with native_function_manager(f):
                    return DispatcherSignature.from_schema(f.func).decl()
            expected_schemas_str = '\n'.join([create_decl(f) for f in funcs])
            missing_kernels_err_msg += f'\n{class_name} is missing a kernel definition for {expected_name}. We found {actual_overload_count} kernel(s) with that name,\nbut expected {expected_overload_count} kernel(s). The expected function schemas for the missing operator are:\n{expected_schemas_str}\n\n'
    assert missing_kernels_err_msg == '', missing_kernels_err_msg