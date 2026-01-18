import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def sparse_prod(x, axis=None, keepdims=False, dtype=None, out=None):
    return x.prod(axis=axis, keepdims=keepdims, dtype=dtype, out=out)