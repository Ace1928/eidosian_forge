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
def make_tol_arg(kwarg_type, inp):
    if kwarg_type == 'none':
        return None
    if kwarg_type == 'float':
        return 1.0
    assert kwarg_type == 'tensor'
    return torch.ones(inp.shape[:-2], device=device)