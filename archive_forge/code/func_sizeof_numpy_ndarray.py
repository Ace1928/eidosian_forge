from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register(np.ndarray)
def sizeof_numpy_ndarray(x):
    if 0 in x.strides:
        xs = x[tuple((slice(None) if s != 0 else slice(1) for s in x.strides))]
        return xs.nbytes
    return int(x.nbytes)