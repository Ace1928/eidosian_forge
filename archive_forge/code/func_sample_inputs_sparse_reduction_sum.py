import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def sample_inputs_sparse_reduction_sum(op_info, device, dtype, requires_grad, layout, **kwargs):
    """Sample inputs for sum on sparse tensors."""
    yield from _sample_inputs_sparse(sample_inputs_sparse_reduction, _maybe_failing_sample_inputs_sparse_reduction_sum, _validate_sample_input_sparse_reduction, op_info, device, dtype, requires_grad, layout, **kwargs)