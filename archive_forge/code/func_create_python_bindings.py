import itertools
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.python import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.gen import cpp_string, parse_native_yaml, parse_tags_yaml
from torchgen.model import (
from torchgen.utils import FileManager, split_name_params
from torchgen.yaml_utils import YamlLoader
from .gen_trace_type import should_trace
def create_python_bindings(fm: FileManager, pairs: Sequence[PythonSignatureNativeFunctionPair], pred: Callable[[NativeFunction], bool], module: Optional[str], filename: str, *, method: bool, symint: bool=True) -> None:
    """Generates Python bindings to ATen functions"""
    py_methods: List[str] = []
    ops_headers: List[str] = []
    py_method_defs: List[str] = []
    py_forwards: List[str] = []
    grouped = group_filter_overloads(pairs, pred)
    for name in sorted(grouped.keys(), key=str):
        overloads = grouped[name]
        py_methods.append(method_impl(name, module, overloads, method=method, symint=symint))
        py_method_defs.append(method_def(name, module, overloads, method=method))
        py_forwards.extend(forward_decls(name, overloads, method=method))
        ops_headers.append(f'#include <ATen/ops/{name.base}.h>')
    fm.write_with_template(filename, filename, lambda: {'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/{filename}', 'ops_headers': ops_headers, 'py_forwards': py_forwards, 'py_methods': py_methods, 'py_method_defs': py_method_defs})