from __future__ import absolute_import
from functools import partial
from collections import OrderedDict
import warnings
from .wrap_util import unary_to_nary
from .builtins import tuple as atuple
from .core import make_vjp as _make_vjp, make_jvp as _make_jvp
from .extend import primitive, defvjp_argnum, vspace
import autograd.numpy as np
def gradfun(*args, **kwargs):
    bindings = sig.bind(*args, **kwargs)
    args = lambda dct: tuple(dct[var_pos[0]]) if var_pos else ()
    kwargs = lambda dct: todict(dct[var_kwd[0]]) if var_kwd else {}
    others = lambda dct: tuple((dct[argname] for argname in argnames if argname not in var_kwd + var_pos))
    newfun = lambda dct: fun(*others(dct) + args(dct), **kwargs(dct))
    argdict = apply_defaults(bindings.arguments)
    grad_dict = grad(newfun)(dict(argdict))
    return OrderedDict(((argname, grad_dict[argname]) for argname in argdict))