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
def error_inputs_lstsq(op_info, device, **kwargs):
    zero_d = torch.randn((), device=device)
    yield ErrorInput(SampleInput(zero_d, args=(zero_d,)), error_type=RuntimeError, error_regex='at least 2 dimensions')