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
def error_inputs_kthvalue(op_info, device, **kwargs):
    t = make_tensor(10, dtype=torch.float32, device=device)
    indices = torch.empty((), device=device, dtype=torch.long)
    yield ErrorInput(SampleInput(t, 5, out=(t, indices)), error_regex='unsupported operation')
    k_out_of_range_err = 'selected number k out of range for dimension'
    yield ErrorInput(SampleInput(torch.randn(2, 2, device=device), 3, 0), error_regex=k_out_of_range_err)
    yield ErrorInput(SampleInput(torch.randn(2, 2, device=device), 3), error_regex=k_out_of_range_err)
    yield ErrorInput(SampleInput(torch.tensor(2, device=device), 3), error_regex=k_out_of_range_err)