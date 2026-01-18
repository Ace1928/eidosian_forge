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
def gen_has_torch_function_check(name: BaseOperatorName, module: Optional[str], *, noarg: bool, method: bool) -> str:
    if noarg:
        if method:
            return f'if(check_has_torch_function(self_)) {{\n  return handle_torch_function(self_, "{name}");\n}}\n'
        else:
            return ''
    self_ = 'self_' if method else 'nullptr'
    namespace = {'torch': 'THPVariableFunctionsModule', 'torch.nn': 'THPNNVariableFunctionsModule', 'torch.fft': 'THPFFTVariableFunctionsModule', 'torch.linalg': 'THPLinalgVariableFunctionsModule', 'torch.nested': 'THPNestedVariableFunctionsModule', 'torch.sparse': 'THPSparseVariableFunctionsModule', 'torch.special': 'THPSpecialVariableFunctionsModule'}[module] if module else 'THPVariableClass'
    return f'if(_r.has_torch_function()) {{\n  return handle_torch_function(_r, {self_}, args, kwargs, {namespace}, "{module or 'torch.Tensor'}");\n}}\n'