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
def method_def(name: BaseOperatorName, module: Optional[str], overloads: Sequence[PythonSignatureNativeFunctionPair], *, method: bool) -> str:
    """
    Generate method def entry.
    """
    pycname = get_pycname(name)
    if name.dunder_method:
        pycname = f'TypeError_to_NotImplemented_<{pycname}>'
    if is_noarg(overloads):
        flags = 'METH_NOARGS' if method else 'METH_VARARGS | METH_KEYWORDS'
    else:
        pycname = f'castPyCFunctionWithKeywords({pycname})'
        flags = 'METH_VARARGS | METH_KEYWORDS'
    if module == 'torch':
        flags += ' | METH_STATIC'
    return f'{{"{name}", {pycname}, {flags}, NULL}},'