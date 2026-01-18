import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
def sample_inputs_numpy_mul(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg(4, low=0.9, high=2), args=(make_arg(3, 4, low=0.9, high=2),))