import unittest
from functools import partial
from itertools import product
from typing import Callable, List, Tuple
import numpy
import torch
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
def error_inputs_exponential_window(op_info, device, **kwargs):
    yield from error_inputs_window(op_info, device, **kwargs)
    yield ErrorInput(SampleInput(3, tau=-1, dtype=torch.float32, device=device, **kwargs), error_type=ValueError, error_regex='Tau must be positive, got: -1 instead.')
    yield ErrorInput(SampleInput(3, center=1, sym=True, dtype=torch.float32, device=device), error_type=ValueError, error_regex='Center must be None for symmetric windows')