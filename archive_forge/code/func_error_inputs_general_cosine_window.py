import unittest
from functools import partial
from itertools import product
from typing import Callable, List, Tuple
import numpy
import torch
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
def error_inputs_general_cosine_window(op_info, device, **kwargs):
    yield from error_inputs_window(op_info, device, a=[0.54, 0.46], **kwargs)
    yield ErrorInput(SampleInput(3, a=None, dtype=torch.float32, device=device, **kwargs), error_type=TypeError, error_regex='Coefficients must be a list/tuple')
    yield ErrorInput(SampleInput(3, a=[], dtype=torch.float32, device=device, **kwargs), error_type=ValueError, error_regex='Coefficients cannot be empty')