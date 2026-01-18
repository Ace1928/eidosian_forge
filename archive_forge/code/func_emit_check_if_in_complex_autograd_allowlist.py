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
def emit_check_if_in_complex_autograd_allowlist() -> List[str]:
    body: List[str] = []
    if base_name in GRADIENT_IMPLEMENTED_FOR_COMPLEX:
        return body
    for arg in differentiable_outputs:
        name = arg.name
        if arg.cpp_type == 'at::Tensor' or arg.cpp_type in TENSOR_LIST_LIKE_CTYPES:
            body.append(f'throw_error_for_complex_autograd({name}, "{base_name}");')
    return body