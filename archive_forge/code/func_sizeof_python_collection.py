from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register(list)
@sizeof.register(tuple)
@sizeof.register(set)
@sizeof.register(frozenset)
def sizeof_python_collection(seq):
    num_items = len(seq)
    num_samples = 10
    if num_items > num_samples:
        if isinstance(seq, (set, frozenset)):
            samples = itertools.islice(seq, num_samples)
        else:
            samples = random.sample(seq, num_samples)
        return sys.getsizeof(seq) + int(num_items / num_samples * sum(map(sizeof, samples)))
    else:
        return sys.getsizeof(seq) + sum(map(sizeof, seq))