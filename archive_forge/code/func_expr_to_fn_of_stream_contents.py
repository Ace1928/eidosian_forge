from collections import namedtuple
import param
from .. import (
from ..plotting.util import initialize_dynamic
from ..streams import Derived, Stream
from . import AdjointLayout, ViewableTree
from .operation import OperationCallable
def expr_to_fn_of_stream_contents(expr, nkdims):

    def eval_expr(expr, kdim_values, stream_values):
        if isinstance(expr, Expr):
            fn = expr.fn
            args = [eval_expr(arg, kdim_values, stream_values) for arg in expr.args]
            kwargs_list = [eval_expr(kwarg, kdim_values, stream_values) for kwarg in expr.kwargs]
            kwargs = {}
            for kwargs_el in kwargs_list:
                kwargs.update(**eval_expr(kwargs_el, kdim_values, stream_values))
            if isinstance(fn, param.ParameterizedFunction):
                kwargs = {k: v for k, v in kwargs.items() if k in fn.param}
            return fn(*args, **kwargs)
        elif isinstance(expr, StreamIndex):
            return stream_values[expr.index]
        elif isinstance(expr, KDimIndex):
            return kdim_values[expr.index]
        elif isinstance(expr, dict):
            return {k: eval_expr(v, kdim_values, stream_values) for k, v in expr.items()}
        elif isinstance(expr, (list, tuple)):
            return type(expr)([eval_expr(v, kdim_values, stream_values) for v in expr])
        else:
            return expr

    def expr_fn(*args):
        kdim_values = args[:nkdims]
        stream_values = args[nkdims:]
        return eval_expr(expr, kdim_values, stream_values)
    return expr_fn