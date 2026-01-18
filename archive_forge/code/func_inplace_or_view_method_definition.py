from typing import Dict, List, Optional, Sequence, Tuple
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import FileManager
from .context import with_native_function_with_differentiability_info
from .gen_trace_type import (
@with_native_function_with_differentiability_info
def inplace_or_view_method_definition(fn: NativeFunctionWithDifferentiabilityInfo) -> Optional[str]:
    f = fn.func
    if get_view_info(f) is None and (not modifies_arguments(f) or len(f.func.returns) == 0):
        return None
    return METHOD_DEFINITION.substitute(return_type=cpp.returns_type(f.func.returns, symint=True).cpp_type(), type_wrapper_name=type_wrapper_name(f), formals=gen_formals(f), type_definition_body=emit_inplace_or_view_body(fn))