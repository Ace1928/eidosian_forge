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
def guard_for(arg: SavedAttribute) -> Optional[str]:
    assert info is not None
    if has_tensorlist_arg and (not is_inplace_foreach):
        return None
    if 'backward' in info.name:
        return None
    if len(args_with_derivatives) <= 1:
        return None
    if arg.nctype.type != BaseCType(tensorT):
        return None
    used_in = [d for d in info.derivatives if arg in d.saved_inputs]
    assert len(used_in) > 0
    if len(used_in) != 1:
        return None
    derivative = used_in[0]
    if len(derivative.var_names) != 1:
        wrap_opt_if_start = derivative.formula.find(f'wrap_opt_if({arg.nctype.name}')
        if wrap_opt_if_start == -1:
            return None
        wrap_opt_if_match = re.match(f'wrap_opt_if\\({arg.nctype.name},(.*?)\\)', derivative.formula[wrap_opt_if_start:])
        assert wrap_opt_if_match is not None
        condition_slice = slice(len(f'wrap_opt_if\\({arg.nctype.name},'), -1)
        wrap_opt_if_condition = wrap_opt_if_match.group(0)[condition_slice].strip()
        wrap_opt_if_condition = re.sub('grad_input_mask\\[(\\d+)\\]', 'grad_fn->should_compute_output(\\1)', wrap_opt_if_condition)
        return f'{wrap_opt_if_condition}'
    derivative_var_name = derivative.var_names[0]
    for edge_off, a in enumerate(args_with_derivatives):
        if a.name == derivative_var_name:
            break
    else:
        raise AssertionError()
    return f'grad_fn->should_compute_output({edge_off})'