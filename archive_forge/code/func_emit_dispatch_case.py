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
def emit_dispatch_case(overload: PythonSignatureGroup, namedtuple_typenames: Dict[str, str], *, symint: bool=True) -> str:
    """
    Emit dispatch code for a single parsed signature. This corresponds to either
    a single native function, or a pair that differ only in output params. In the
    latter case, a single python signature is used for both and dispatching
    switches on the presence/absence of passed output args.
    """
    if overload.outplace is not None:
        return PY_VARIABLE_OUT.substitute(out_idx=overload.signature.output_idx(), call_dispatch=emit_single_dispatch(overload.signature, overload.base, namedtuple_typenames, symint=symint), call_dispatch_out=emit_single_dispatch(overload.signature, overload.outplace, namedtuple_typenames, symint=symint))
    else:
        return emit_single_dispatch(overload.signature, overload.base, namedtuple_typenames, symint=symint)