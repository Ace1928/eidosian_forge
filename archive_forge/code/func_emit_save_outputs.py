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
def emit_save_outputs() -> str:
    if is_out_fn:
        return ''
    if info is not None and info.has_derivatives:
        stmts = save_variables(info.all_saved_outputs, True)
        if len(stmts) == 0:
            return ''
        if not is_inplace_foreach:
            return CONDITIONAL.substitute(cond='grad_fn', statements=stmts)
        else:
            return LOOP_OVER_VECTOR_OF_GRAD_FNS.substitute(preamble='', statements=stmts)
    return ''