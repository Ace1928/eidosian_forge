from collections import namedtuple
import param
from .. import (
from ..plotting.util import initialize_dynamic
from ..streams import Derived, Stream
from . import AdjointLayout, ViewableTree
from .operation import OperationCallable
def expr_fn(*args):
    kdim_values = args[:nkdims]
    stream_values = args[nkdims:]
    return eval_expr(expr, kdim_values, stream_values)