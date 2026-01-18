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
def error_inputs_complex(op_info, device, is_ref=False, **kwargs):
    make_arg = partial(make_tensor, dtype=torch.float32, device=device)
    if is_ref:
        error_float = 'Expected both inputs to be Half, Float or Double tensors but got torch.float32 and torch.int32'
        error_dtype = 'Expected object of scalar type torch.float32 but got scalar type torch.float64 for second argument'
        error_out = 'Expected out tensor to have dtype torch.complex128 but got torch.complex64 instead'
    else:
        error_float = 'Expected both inputs to be Half, Float or Double tensors but got Float and Int'
        error_dtype = 'Expected object of scalar type Float but got scalar type Double for second argument'
        error_out = "Expected object of scalar type ComplexDouble but got scalar type ComplexFloat for argument 'out'"
    yield ErrorInput(SampleInput(make_arg(M, S), make_arg(M, S, dtype=torch.int)), error_type=RuntimeError, error_regex=error_float)
    yield ErrorInput(SampleInput(make_arg(M, S), make_arg(M, S, dtype=torch.float64)), error_type=RuntimeError, error_regex=error_dtype)
    yield ErrorInput(SampleInput(make_arg(M, S, dtype=torch.float64), make_arg(M, S, dtype=torch.float64), out=make_arg(M, S, dtype=torch.complex64)), error_type=RuntimeError, error_regex=error_out)