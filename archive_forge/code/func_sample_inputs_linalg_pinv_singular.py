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
def sample_inputs_linalg_pinv_singular(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function produces factors `a` and `b` to generate inputs of the form `a @ b.t()` to
    test the backward method of `linalg_pinv`. That way we always preserve the rank of the
    input no matter the perturbations applied to it by the gradcheck.
    Note that `pinv` is Frechet-differentiable in a rank-preserving neighborhood.
    """
    batches = [(), (0,), (2,), (1, 1)]
    size = [0, 3, 50]
    for batch, m, n in product(batches, size, size):
        for k in range(min(3, m, n)):
            a = torch.rand(*batch, m, k, device=device, dtype=dtype).qr().Q.requires_grad_(requires_grad)
            b = torch.rand(*batch, n, k, device=device, dtype=dtype).qr().Q.requires_grad_(requires_grad)
            yield SampleInput(a, args=(b,))