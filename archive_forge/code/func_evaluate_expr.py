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
def evaluate_expr(self, output_graph=None):
    try:
        return guard_scalar(self.sym_num)
    except GuardOnDataDependentSymNode as e:
        raise UserError(UserErrorType.ANTI_PATTERN, f'Consider annotating your code using torch._constrain_as_*(). {str(e)}', case_name='constrain_as_size_example')