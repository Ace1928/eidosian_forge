import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
def sample_inputs_foreach(self, device, dtype, N, *, noncontiguous=False, same_size=False, low=None, high=None, zero_size: bool, requires_grad: bool, intersperse_empty_tensors: bool=False):
    if zero_size:
        return [torch.empty(0, dtype=dtype, device=device) for _ in range(N)]
    if same_size:
        return [make_tensor((N, N), dtype=dtype, device=device, noncontiguous=noncontiguous, low=low, high=high, requires_grad=requires_grad) for _ in range(N)]
    else:
        return [torch.empty(0, dtype=dtype, device=device, requires_grad=requires_grad) if (i % 3 == 0 or i >= N - 2) and intersperse_empty_tensors else make_tensor((N - i, N - i), dtype=dtype, device=device, noncontiguous=noncontiguous, low=low, high=high, requires_grad=requires_grad) for i in range(N)]