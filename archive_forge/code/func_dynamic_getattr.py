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
def dynamic_getattr(self, tx, name):
    if not self.source:
        raise NotImplementedError()
    scope = {'L': tx.output.local_scope, 'G': tx.output.global_scope}
    try:
        _input_associated_real_value = eval(self.source.name(), scope)
    except Exception as exc:
        raise NotImplementedError() from exc
    if _input_associated_real_value is None:
        raise NotImplementedError()
    if object_has_getattribute(_input_associated_real_value):
        raise NotImplementedError()
    if get_custom_getattr(_input_associated_real_value):
        raise NotImplementedError()
    real_value = getattr(_input_associated_real_value, name)
    if callable(real_value):
        raise NotImplementedError()
    from ..guards import GuardBuilder
    from .builder import VariableBuilder
    attr_source = AttrSource(self.source, name)
    install_guard(attr_source.make_guard(GuardBuilder.HASATTR))
    return VariableBuilder(tx, attr_source)(real_value)