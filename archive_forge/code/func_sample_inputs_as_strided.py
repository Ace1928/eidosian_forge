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
def sample_inputs_as_strided(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    test_cases = (((1,), (1,), (1,), 0), ((3, 3), (2, 2), (1, 2), 0), ((3, 3), (2, 2), (1, 2), 1), ((16,), (2, 2, 2, 2), (1, 1, 1, 1), 0), ((16,), (2, 1, 1, 2), (1, 7, 7, 1), 0))
    for input_shape, output_shape, stride, storage_offset in test_cases:
        input_t = make_arg(input_shape)
        kwargs = dict(storage_offset=storage_offset)
        yield SampleInput(input_t, args=(output_shape, stride), kwargs=kwargs)