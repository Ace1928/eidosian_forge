import itertools
import random
import unittest
from functools import partial
from itertools import chain, product
from typing import Iterable, List
import numpy as np
from numpy import inf
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import PythonRefInfo, ReductionPythonRefInfo
def sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates invertible inputs for linear algebra ops
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (1, 1) - 1x1 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    make_fn = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(make_fn, dtype=dtype, device=device, requires_grad=requires_grad)
    batches = [(), (0,), (2,), (1, 1)]
    ns = [5, 0]
    for batch, n in product(batches, ns):
        yield SampleInput(make_arg(*batch, n, n))