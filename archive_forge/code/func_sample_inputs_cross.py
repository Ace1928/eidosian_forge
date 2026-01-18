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
def sample_inputs_cross(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, 3)), args=(make_arg((S, 3)),))
    yield SampleInput(make_arg((S, 3, S)), args=(make_arg((S, 3, S)),), kwargs=dict(dim=1))
    yield SampleInput(make_arg((1, 3)), args=(make_arg((S, 3)),), kwargs=dict(dim=-1))