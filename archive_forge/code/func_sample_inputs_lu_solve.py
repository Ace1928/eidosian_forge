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
def sample_inputs_lu_solve(op_info, device, dtype, requires_grad=False, **kwargs):
    """Samples the inputs for both linalg.lu_solve and lu_solve"""
    make_fn = make_fullrank_matrices_with_distinct_singular_values
    make_a = partial(make_fn, dtype=dtype, device=device)
    make_b = partial(make_tensor, dtype=dtype, device=device)

    def clone(X, requires_grad):
        Y = X.clone()
        Y.requires_grad_(requires_grad)
        return Y
    is_linalg_lu_solve = op_info.name == 'linalg.lu_solve'
    batches = ((), (0,), (2,))
    ns = (3, 1, 0)
    nrhs = (4, 1, 0)
    for n, batch, rhs in product(ns, batches, nrhs):
        A = make_a(*batch + (n, n))
        LU, pivots = torch.linalg.lu_factor(A)
        B = make_b(batch + (n, rhs))
        grads = (False,) if not requires_grad else (True, False)
        for LU_grad, B_grad in product(grads, grads):
            if requires_grad and (not LU_grad) and (not B_grad):
                continue
            if is_linalg_lu_solve:
                for adjoint, left in product((True, False), repeat=2):
                    yield SampleInput(clone(LU, LU_grad), args=(pivots, clone(B if left else B.mT, B_grad)), kwargs=dict(adjoint=adjoint, left=left))
            else:
                yield SampleInput(clone(B, B_grad), args=(clone(LU, LU_grad), pivots))