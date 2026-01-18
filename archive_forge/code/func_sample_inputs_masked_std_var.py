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
def sample_inputs_masked_std_var(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked std/var."""
    kwargs['supports_multiple_dims'] = op_info.supports_multiple_dims
    from torch.testing._internal.common_methods_invocations import sample_inputs_std_var

    def masked_samples():
        for sample_input in sample_inputs_std_var(op_info, device, dtype, requires_grad, **kwargs):
            if len(sample_input.args) and isinstance(sample_input.args[0], bool):
                continue
            for mask in _generate_masked_op_mask(sample_input.input.shape, device, **kwargs):
                sample_input_args, sample_input_kwargs = (sample_input.args, dict(mask=mask, **sample_input.kwargs))
                yield SampleInput(sample_input.input.detach().requires_grad_(requires_grad), args=sample_input_args, kwargs=sample_input_kwargs)
                if not requires_grad and dtype.is_floating_point and (sample_input.input.ndim == 2) and (mask is not None) and (mask.shape == sample_input.input.shape):
                    for v in [torch.inf, -torch.inf, torch.nan]:
                        t = sample_input.input.detach()
                        t.diagonal(0, -2, -1).fill_(v)
                        yield SampleInput(t.requires_grad_(requires_grad), args=sample_input_args, kwargs=sample_input_kwargs)
    for sample_input in masked_samples():
        correction = sample_input.kwargs.get('correction')
        if correction is None:
            correction = int(sample_input.kwargs.get('unbiased', True))
        dim = sample_input.kwargs.get('dim', None)
        if sample_input.kwargs.get('mask') is None:
            orig_count = torch.masked.sum(torch.ones(sample_input.input.shape, dtype=torch.int64), dim, keepdim=True)
        else:
            inmask = torch.masked._input_mask(sample_input.input, *sample_input.args, **sample_input.kwargs)
            orig_count = torch.masked.sum(inmask.new_ones(sample_input.input.shape, dtype=torch.int64), dim, keepdim=True, mask=inmask)
        if orig_count.min() <= correction + 1:
            continue
        yield sample_input