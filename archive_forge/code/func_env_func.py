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
def env_func(kv: Tuple[BaseOperatorName, List[PythonSignatureNativeFunctionPair]]) -> Dict[str, List[str]]:
    name, fn_pairs = kv
    return {'ops_headers': [f'#include <ATen/ops/{name.base}.h>'], 'py_forwards': list(forward_decls(name, fn_pairs, method=method)), 'py_methods': [method_impl(name, module, fn_pairs, method=method, symint=symint)], 'py_method_defs': [method_def(name, module, fn_pairs, method=method)]}