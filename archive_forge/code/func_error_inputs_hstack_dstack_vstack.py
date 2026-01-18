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
def error_inputs_hstack_dstack_vstack(op, device):
    make_arg = partial(make_tensor, dtype=torch.int32, device=device, requires_grad=False)
    tensor_shapes = (((S,), (S, S, S, S), (S,)),)
    for s1, s2, s3 in tensor_shapes:
        tensors = (make_arg(s1), make_arg(s2), make_arg(s3))
        yield ErrorInput(SampleInput(tensors), error_regex='Tensors must have same number of dimensions')
    yield ErrorInput(SampleInput(()), error_regex='expects a non-empty TensorList')