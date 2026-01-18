from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
def sample_inputs_arange(op, device, dtype, requires_grad, **kwargs):
    int_samples = ((-1, 2, 2), (2, -3, -1), (1, 1, 1), (1, 1, -1), (0, -8, -4), (1, 5, 2), (False, True, True), (0, 1, None), (None, 3, None))

    def to_float(start, end, step):
        start = start + 0.1 if start is not None else None
        end = end + 0.1
        step = float(step) if step is not None else None
        return (start, end, step)
    float_samples = ((0.0, -8.0 - 1e-06, -4.0), (1.0, 5.0 + 1e-06, 2.0), (0.0, -8.0, -4.0), (1.0, 5.0, 2.0), *(to_float(start, end, step) for start, end, step in int_samples))
    large_samples = ((0, 10000, None),)
    samples = int_samples + float_samples
    if dtype not in (torch.int8, torch.uint8):
        samples += large_samples
    for start, end, step in samples:
        if start is None:
            assert step is None
            yield SampleInput(end, kwargs={'dtype': dtype, 'device': device})
            yield SampleInput(0, kwargs={'end': end, 'dtype': dtype, 'device': device})
        elif step is None:
            yield SampleInput(start, args=(end,), kwargs={'dtype': dtype, 'device': device})
        else:
            yield SampleInput(start, args=(end, step), kwargs={'dtype': dtype, 'device': device})
    yield SampleInput(2)
    yield SampleInput(1, args=(3, 1))