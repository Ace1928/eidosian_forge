from __future__ import absolute_import
from functools import partial
from collections import OrderedDict
import warnings
from .wrap_util import unary_to_nary
from .builtins import tuple as atuple
from .core import make_vjp as _make_vjp, make_jvp as _make_jvp
from .extend import primitive, defvjp_argnum, vspace
import autograd.numpy as np
def grad_named(fun, argname):
    """Takes gradients with respect to a named argument.
       Doesn't work on *args or **kwargs."""
    arg_index = _getargspec(fun).args.index(argname)
    return grad(fun, arg_index)