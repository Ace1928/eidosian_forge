import unittest
from functools import partial
from itertools import product
from typing import Callable, List, Tuple
import numpy
import torch
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
def sample_inputs_window(op_info, device, dtype, requires_grad, *args, **kwargs):
    """Base function used to create sample inputs for windows.

    For additional required args you should use *args, as well as **kwargs for
    additional keyword arguments.
    """
    for size, sym in product(range(6), (True, False)):
        yield SampleInput(size, *args, sym=sym, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)