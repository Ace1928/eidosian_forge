from __future__ import absolute_import
from functools import partial
from collections import OrderedDict
import warnings
from .wrap_util import unary_to_nary
from .builtins import tuple as atuple
from .core import make_vjp as _make_vjp, make_jvp as _make_jvp
from .extend import primitive, defvjp_argnum, vspace
import autograd.numpy as np
def make_ggnvp(f, g=lambda x: 1.0 / 2 * np.sum(x ** 2, axis=-1), f_argnum=0):
    """Builds a function for evaluating generalized-Gauss-Newton-vector products
    at a point. Slightly more expensive than mixed-mode."""

    @unary_to_nary
    def _make_ggnvp(f, x):
        f_vjp, f_x = _make_vjp(f, x)
        g_hvp, grad_g_x = _make_vjp(grad(g), f_x)
        f_jvp, _ = _make_vjp(f_vjp, vspace(grad_g_x).zeros())

        def ggnvp(v):
            return f_vjp(g_hvp(f_jvp(v)))
        return ggnvp
    return _make_ggnvp(f, f_argnum)