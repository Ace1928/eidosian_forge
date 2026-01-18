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
def make_error_inputs_elementwise_binary(error_inputs_func):

    def error_inputs_func_wrapper(op, device, **kwargs):
        if error_inputs_func is not None:
            yield from error_inputs_func(op, device, **kwargs)
        if not op.supports_rhs_python_scalar:
            si = SampleInput(torch.tensor((1, 2, 3), device=device), args=(2,))
            yield ErrorInput(si, error_type=Exception, error_regex='')
        if not op.supports_one_python_scalar:
            si = SampleInput(2, args=(torch.tensor((1, 2, 3), device=device),))
            yield ErrorInput(si, error_type=Exception, error_regex='')
        if not kwargs.get('skip_two_python_scalars', False) and (not op.supports_two_python_scalars):
            si = SampleInput(2, args=(3,))
            yield ErrorInput(si, error_type=Exception, error_regex='')
    return error_inputs_func_wrapper