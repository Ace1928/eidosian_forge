import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def sparse_random_uniform(low=0.0, high=1.0, size=None, dtype=None, **kwargs):

    def rvs(nnz):
        return do('random.uniform', low, high, (nnz,), dtype=dtype, like='numpy')
    return do('random', size, data_rvs=rvs, **kwargs, like='sparse')