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
def sample_inputs_linalg_cholesky_inverse(op_info, device, dtype, requires_grad=False, **kwargs):
    from torch.testing._internal.common_utils import random_well_conditioned_matrix
    single_well_conditioned_matrix = random_well_conditioned_matrix(S, S, dtype=dtype, device=device)
    batch_well_conditioned_matrices = random_well_conditioned_matrix(2, S, S, dtype=dtype, device=device)
    single_pd = single_well_conditioned_matrix @ single_well_conditioned_matrix.mH
    batch_pd = batch_well_conditioned_matrices @ batch_well_conditioned_matrices.mH
    inputs = (torch.zeros(0, 0, dtype=dtype, device=device), torch.zeros(0, 2, 2, dtype=dtype, device=device), single_pd, batch_pd)
    test_cases = (torch.linalg.cholesky(a, upper=False) for a in inputs)
    for l in test_cases:
        l.requires_grad = requires_grad
        yield SampleInput(l)
        yield SampleInput(l.detach().clone().requires_grad_(requires_grad), kwargs=dict(upper=False))
        u = l.detach().clone().mT.contiguous().requires_grad_(requires_grad)
        yield SampleInput(u, kwargs=dict(upper=True))