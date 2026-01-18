import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def dask_eye_wrapper(fn):

    @functools.wraps(fn)
    def numpy_like(N, M=None, **kwargs):
        return fn(N, M=M, **kwargs)
    return numpy_like