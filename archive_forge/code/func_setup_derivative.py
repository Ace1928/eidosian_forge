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
def setup_derivative(differentiable_inputs: List[DifferentiableInput]) -> List[str]:
    body: List[str] = []
    if is_out_fn:
        body.append(DECLARE_GRAD_FN.substitute(op='Node'))
        body.append(SETUP_NONE_REQUIRES_GRAD.substitute(base_name=base_name, args_to_check=[arg.name for arg in differentiable_inputs]))
        body.append(SETUP_NONE_REQUIRES_GRAD.substitute(base_name=base_name, args_to_check=[arg.name for arg in differentiable_outputs]))
        return body
    op = info.op if info is not None and info.has_derivatives else 'NotImplemented'
    setup = []
    if not is_inplace_foreach:
        setup.extend(ASSIGN_GRAD_FN.substitute(op=op, op_ctor='' if info is not None and info.has_derivatives else f'"{cpp.name(f.func)}"', args_with_derivatives=[arg.name for arg in args_with_derivatives]).split('\n'))
    else:
        list_like_arg = 'self'
        args = [arg.name for arg in args_with_derivatives]
        for i, arg in enumerate(args):
            if is_inplace_foreach and info is not None:
                if arg in refargname2inplace_foreacharg:
                    foreach_arg = refargname2inplace_foreacharg[arg]
                    args[i] = foreach_arg.name + ('[i]' if isinstance(foreach_arg.type, ListType) else '')
            elif arg == list_like_arg:
                args[i] = arg + '[i]'
        setup.extend(ASSIGN_VECTOR_OF_GRAD_FN.substitute(op=op, op_ctor='' if info is not None and info.has_derivatives else f'"{cpp.name(f.func)}"', args_with_derivatives=args, irange=f'{list_like_arg}.size()').split('\n'))
    setup.extend(emit_save_inputs())
    body.extend(emit_check_no_requires_grad(differentiable_inputs, args_with_derivatives))
    declare_grad_fn_template = DECLARE_GRAD_FN if not is_inplace_foreach else DECLARE_VECTOR_OF_GRAD_FN
    body.append(declare_grad_fn_template.substitute(op=op))
    body.append(SETUP_DERIVATIVE.substitute(setup=setup))
    return body