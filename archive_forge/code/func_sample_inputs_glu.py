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
def sample_inputs_glu(self, device, dtype, requires_grad, **kwargs):
    features_options = [[2], [2, 4], [8, 8], [3, 6, 8], [1, 4, 6, 7]]
    batch_options: List[List[int]] = [[], [0], [8], [2, 3]]
    create_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-2, high=2)
    for features, batch_shape in itertools.product(features_options, batch_options):
        ndim = len(features) + len(batch_shape)
        for dim in range(ndim):
            input_tensor = create_tensor(batch_shape + features)
            dim_size = input_tensor.size(dim)
            if dim_size > 0 and dim_size % 2 == 0:
                yield SampleInput(input_tensor, dim)