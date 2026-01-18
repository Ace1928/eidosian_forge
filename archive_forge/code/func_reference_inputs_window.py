import unittest
from functools import partial
from itertools import product
from typing import Callable, List, Tuple
import numpy
import torch
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
def reference_inputs_window(op_info, device, dtype, requires_grad, *args, **kwargs):
    """Reference inputs function to use for windows which have a common signature, i.e.,
    window size and sym only.

    Implement other special functions for windows that have a specific signature.
    See exponential and gaussian windows for instance.
    """
    yield from sample_inputs_window(op_info, device, dtype, requires_grad, *args, **kwargs)
    cases = (8, 16, 32, 64, 128, 256)
    for size in cases:
        yield SampleInput(size, sym=False)
        yield SampleInput(size, sym=True)