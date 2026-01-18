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
def error_inputs_rot90(op_info, device, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    err_msg1 = 'expected total rotation dims'
    s1 = SampleInput(make_arg((S, S)), dims=(0,))
    yield ErrorInput(s1, error_regex=err_msg1)
    err_msg2 = 'expected total dims >= 2'
    s2 = SampleInput(make_arg((S,)))
    yield ErrorInput(s2, error_regex=err_msg2)
    err_msg3 = 'expected rotation dims to be different'
    s3 = SampleInput(make_arg((S, S)), dims=(1, 1))
    yield ErrorInput(s3, error_regex=err_msg3)