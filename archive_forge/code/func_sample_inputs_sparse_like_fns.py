import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def sample_inputs_sparse_like_fns(op_info, device, dtype, requires_grad, layout, **kwargs):
    """Sample inputs for like-functions on sparse tensors."""
    yield from _sample_inputs_sparse(_sample_inputs_sparse_like_fns, _maybe_failing_sample_inputs_sparse_like_fns, _validate_sample_input_sparse_like_fns, op_info, device, dtype, requires_grad, layout, **kwargs)