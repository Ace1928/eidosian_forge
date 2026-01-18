import functools
import inspect
import operator
import types
from typing import Dict, List
import sympy
import torch._numpy as tnp
import torch.fx
import torch.random
from torch._dynamo import compiled_autograd
from torch.fx.experimental.symbolic_shapes import (
from .. import config, variables
from .._trace_wrapped_higher_order_op import trace_wrapped
from ..exc import unimplemented, UserError, UserErrorType
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource
from ..utils import (
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import SizeVariable
def has_bool_key(v):
    if isinstance(v, TensorVariable):
        return v.dtype in (torch.bool, torch.int8)
    elif isinstance(v, TupleVariable):
        return any((has_bool_key(item) for item in v.items))
    else:
        return False