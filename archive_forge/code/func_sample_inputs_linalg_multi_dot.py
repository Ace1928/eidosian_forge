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
def sample_inputs_linalg_multi_dot(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = [[1, 2, 1], [2, 0, 2], [0, 2, 2], [2, 2, 2, 2], [2, 3, 4, 5], [5, 4, 0, 2], [2, 4, 3, 5, 3, 2]]
    for sizes in test_cases:
        tensors = []
        for size in zip(sizes[:-1], sizes[1:]):
            t = make_tensor(size, dtype=dtype, device=device, requires_grad=requires_grad)
            tensors.append(t)
        yield SampleInput(tensors)