from typing import TYPE_CHECKING
import torch
from . import allowed_functions
from .eval_frame import DisableContext, innermost_fn, RunOnlyContext
from .exc import IncorrectUsage
@forbid_in_graph
def mark_static_address(t, guard=True):
    """
    Marks an input tensor whose data_ptr will not change across multiple calls
    to a dynamo-compiled function. This indicates to cudagraphs that an extra allocation
    is not needed for this input. The data_ptr will be guarded if guard=True. Note:
    Tensors marked in this way will be kept alive until `torch._dynamo.reset()` is called.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f'mark_static_address expects a tensor but recieved {type(t)}')
    if guard:
        t._dynamo_static_input_type = 'guarded'
    else:
        t._dynamo_static_input_type = 'unguarded'