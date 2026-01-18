from functools import partial
from typing import no_type_check, Optional, Tuple
import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import DTensorSpec
@no_type_check
def sync_grad_hook(grad, *, device_handle=None, compute_stream=None):
    if isinstance(grad, AsyncCollectiveTensor):
        if compute_stream is not None:
            with device_handle.stream(compute_stream):
                grad = grad.wait()
        else:
            grad = grad.wait()
    return grad