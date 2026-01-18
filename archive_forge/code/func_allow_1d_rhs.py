import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
@functools.wraps(fn)
def allow_1d_rhs(a, b):
    need_to_convert = ndim(a) != ndim(b)
    if need_to_convert:
        b = reshape(b, (*shape(b), 1))
    x = fn(a, b)
    if need_to_convert:
        x = reshape(x, shape(x)[:-1])
    return x