import unittest
from functools import partial
from itertools import product
from typing import Callable, List, Tuple
import numpy
import torch
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
def reference_inputs_exponential_window(op_info, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_window(op_info, device, dtype, requires_grad, **kwargs)
    cases = ((8, {'center': 4, 'tau': 0.5}), (16, {'center': 8, 'tau': 2.5}), (32, {'center': 16, 'tau': 43.5}), (64, {'center': 20, 'tau': 3.7}), (128, {'center': 62, 'tau': 99}), (256, {'tau': 10}))
    for size, kw in cases:
        yield SampleInput(size, sym=False, **kw)
        kw['center'] = None
        yield SampleInput(size, sym=True, **kw)