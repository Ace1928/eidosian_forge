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
def sample_inputs_meshgrid(op_info: OpInfo, device: torch.device, dtype: torch.dtype, requires_grad: bool, *, variant: str, **kwargs) -> List[SampleInput]:
    if variant == 'variadic':

        def make_inputs(tensors: List[torch.Tensor]) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor, ...]]:
            return tensors
    elif variant == 'list':

        def make_inputs(tensors: List[torch.Tensor]) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor, ...]]:
            return [tensors]
    else:
        raise ValueError(f'Unsupported variant, must be one of {{"variadic", "list"}}. Got "{variant}".')
    SCALAR = torch.Size([])
    VECTOR = torch.Size([3])
    test_cases: List[List[torch.Size]] = [[SCALAR], [VECTOR], [VECTOR, SCALAR], [VECTOR, SCALAR, VECTOR], [VECTOR, SCALAR, VECTOR, SCALAR]]
    for shapes, indexing in itertools.product(test_cases, {'xy', 'ij'}):
        args = make_inputs([make_tensor(shape, dtype=dtype, device=device, requires_grad=requires_grad) for shape in shapes])
        yield SampleInput(*args, indexing=indexing)