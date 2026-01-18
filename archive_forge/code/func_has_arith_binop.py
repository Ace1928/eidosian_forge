import operator
from typing import Dict, List
import torch
from torch._dynamo.source import GetItemSource
from .. import variables
from ..exc import unimplemented, UserError, UserErrorType
from ..guards import GuardBuilder, install_guard
from ..utils import np
from .base import typestr, VariableTracker
def has_arith_binop(num_ty):
    return isinstance(self.value, num_ty) and hasattr(operator, name) and (len(args) == 1) and args[0].is_python_constant()