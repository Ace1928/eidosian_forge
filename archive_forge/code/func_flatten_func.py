from autograd import make_vjp
from autograd.builtins import type
import autograd.numpy as np
def flatten_func(func, example):
    _ex, unflatten = flatten(example)
    _func = lambda _x, *args: flatten(func(unflatten(_x), *args))[0]
    return (_func, unflatten, _ex)