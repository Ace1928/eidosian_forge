import unittest
from collections.abc import Sequence
from functools import partial
from typing import List
import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import tol, toleranceOverride
from torch.testing._internal.common_dtype import (
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.utils import prod_numpy, reference_reduction_numpy
def sample_inputs_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked reduction operators.

    Masked reduction operator is a reduction operator with trailing
    mask optional argument. A mask is a bool tensor with the same
    shape as input or a shape that is broadcastable to input shape.
    """
    kwargs['supports_multiple_dims'] = op_info.supports_multiple_dims
    for sample_input in sample_inputs_reduction(op_info, device, dtype, requires_grad, **kwargs):
        for mask in _generate_masked_op_mask(sample_input.input.shape, device, **kwargs):
            sample_input_args, sample_input_kwargs = (sample_input.args, dict(mask=mask, **sample_input.kwargs))
            yield SampleInput(sample_input.input.detach().requires_grad_(requires_grad), args=sample_input_args, kwargs=sample_input_kwargs)
            if not requires_grad and dtype.is_floating_point and (sample_input.input.ndim == 2) and (mask is not None) and (mask.shape == sample_input.input.shape):
                for v in [torch.inf, -torch.inf, torch.nan]:
                    t = sample_input.input.detach()
                    t.diagonal(0, -2, -1).fill_(v)
                    yield SampleInput(t.requires_grad_(requires_grad), args=sample_input_args, kwargs=sample_input_kwargs)