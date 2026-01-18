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
def sample_inputs_elementwise_unary(op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs):
    if not op_kwargs:
        op_kwargs = {}
    _L = S if kwargs.get('small_inputs_only', False) else L
    low, high = op_info.domain
    is_floating = dtype.is_floating_point or dtype.is_complex
    low = low if low is None or not is_floating else low + op_info._domain_eps
    high = high if high is None or not is_floating else high - op_info._domain_eps
    if op_info.supports_sparse_csr or op_info.supports_sparse_csc or op_info.supports_sparse_bsr or op_info.supports_sparse_bsc:
        yield SampleInput(make_tensor((_L, _L), device=device, dtype=dtype, low=low, high=high, requires_grad=requires_grad), kwargs=op_kwargs)
    else:
        for shape in ((_L,), (1, 0, 3), ()):
            yield SampleInput(make_tensor(shape, device=device, dtype=dtype, low=low, high=high, requires_grad=requires_grad), kwargs=op_kwargs)