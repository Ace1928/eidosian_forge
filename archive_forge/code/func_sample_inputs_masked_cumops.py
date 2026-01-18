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
def sample_inputs_masked_cumops(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked cumsum and cumprod."""
    inputs: List[SampleInput] = []
    for sample_input in sample_inputs_softmax_variant(op_info, device, dtype, requires_grad, **kwargs):
        for mask in _generate_masked_op_mask(sample_input.input.shape, device, **kwargs):
            if type(mask) != torch.Tensor:
                continue
            sample_input_args, sample_input_kwargs = (sample_input.args, dict(mask=mask, **sample_input.kwargs))
            if 'keepdim' in sample_input_kwargs:
                sample_input_kwargs.pop('keepdim')
            if sample_input_args:
                dim = sample_input.args[0]
            else:
                if 'dim' not in sample_input_kwargs:
                    continue
                dim = sample_input_kwargs.pop('dim')
                sample_input_args = (dim,)
            yield SampleInput(sample_input.input.clone().requires_grad_(requires_grad), *sample_input_args, **sample_input_kwargs)