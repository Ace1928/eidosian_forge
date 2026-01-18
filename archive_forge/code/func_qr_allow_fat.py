import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def qr_allow_fat(fn):

    @functools.wraps(fn)
    def numpy_like(a, **kwargs):
        m, n = shape(a)
        if m >= n:
            return fn(a, **kwargs)
        Q, R_sq = fn(a[:, :m])
        R_r = dag(Q) @ a[:, m:]
        R = do('concatenate', (R_sq, R_r), axis=1, like=a)
        return (Q, R)
    return numpy_like