import unittest
from functools import partial
from itertools import product
from typing import Callable, List, Tuple
import numpy
import torch
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_utils import TEST_SCIPY
from torch.testing._internal.opinfo.core import (
def error_inputs_window(op_info, device, *args, **kwargs):
    yield ErrorInput(SampleInput(-1, *args, dtype=torch.float32, device=device, **kwargs), error_type=ValueError, error_regex='requires non-negative window length, got M=-1')
    yield ErrorInput(SampleInput(3, *args, layout=torch.sparse_coo, device=device, dtype=torch.float32, **kwargs), error_type=ValueError, error_regex='is implemented for strided tensors only, got: torch.sparse_coo')
    yield ErrorInput(SampleInput(3, *args, dtype=torch.long, device=device, **kwargs), error_type=ValueError, error_regex='expects float32 or float64 dtypes, got: torch.int64')
    yield ErrorInput(SampleInput(3, *args, dtype=torch.bfloat16, device=device, **kwargs), error_type=ValueError, error_regex='expects float32 or float64 dtypes, got: torch.bfloat16')
    yield ErrorInput(SampleInput(3, *args, dtype=torch.float16, device=device, **kwargs), error_type=ValueError, error_regex='expects float32 or float64 dtypes, got: torch.float16')