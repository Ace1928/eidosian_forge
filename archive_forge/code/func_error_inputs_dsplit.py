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
def error_inputs_dsplit(op_info, device, **kwargs):
    make_arg = partial(make_tensor, dtype=torch.float32, device=device)
    err_msg1 = 'torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with 1 dimensions!'
    yield ErrorInput(SampleInput(make_arg(S), 0), error_regex=err_msg1)
    err_msg2 = f'torch.dsplit attempted to split along dimension 2, but the size of the dimension {S} is not divisible by the split_size 0!'
    yield ErrorInput(SampleInput(make_arg(S, S, S), 0), error_regex=err_msg2)