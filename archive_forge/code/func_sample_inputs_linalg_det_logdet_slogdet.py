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
def sample_inputs_linalg_det_logdet_slogdet(op_info, device, dtype, requires_grad, **kwargs):
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad)
    batches = [(), (0,), (3,)]
    ns = [0, 1, 5]
    is_logdet = op_info.name == 'logdet'
    for batch, n in product(batches, ns):
        shape = batch + (n, n)
        A = make_arg(*shape)
        if is_logdet and (not A.is_complex()) and (A.numel() > 0):
            s = torch.linalg.slogdet(A).sign
            A = A * s.unsqueeze(-1).unsqueeze(-1)
            A.requires_grad_(requires_grad)
        yield SampleInput(A)