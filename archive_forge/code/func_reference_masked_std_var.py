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
def reference_masked_std_var(numpy_fn):
    ref = reference_reduction_numpy(numpy_fn)

    def func(input, dim=None, unbiased=None, *, correction=None, **kwargs):
        ddof = 1
        if unbiased is not None:
            ddof = 1 if unbiased else 0
        if correction is not None:
            ddof = correction
        if isinstance(dim, Sequence):
            dim = tuple(dim)
        return ref(input, dim, ddof=ddof, **kwargs)
    return func