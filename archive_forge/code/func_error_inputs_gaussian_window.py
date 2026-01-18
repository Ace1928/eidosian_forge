import unittest
from functools import partial
from itertools import product
from typing import Callable, List, Tuple
import numpy
import torch
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
def error_inputs_gaussian_window(op_info, device, **kwargs):
    yield from error_inputs_window(op_info, device, std=0.5, **kwargs)
    yield ErrorInput(SampleInput(3, std=-1, dtype=torch.float32, device=device, **kwargs), error_type=ValueError, error_regex='Standard deviation must be positive, got: -1 instead.')