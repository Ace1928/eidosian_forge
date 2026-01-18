import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import (
from torchgen.model import (
from torchgen.utils import FileManager, mapMaybe
from .context import with_native_function_with_differentiability_info_and_key
from .gen_inplace_or_view_type import (
from .gen_trace_type import (
def emit_dispatch_call(f: NativeFunction, input_base: str, unpacked_args: Sequence[str]) -> str:
    """Dispatch call via function in a namespace or method on Tensor."""
    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()
    dispatch_key_set = 'ks & c10::after_autograd_keyset'
    call = CALL_REDISPATCH.substitute(api_name=cpp.name(f.func, faithful_name_for_out_overloads=True, symint_overload=f.func.has_symint()), unpacked_args=[dispatch_key_set] + list(unpacked_args))
    return call