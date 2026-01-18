from __future__ import annotations
import contextlib
import importlib
import numbers
from itertools import chain, product
from numbers import Integral
from operator import getitem
from threading import Lock
import numpy as np
from dask.array.backends import array_creation_dispatch
from dask.array.core import (
from dask.array.creation import arange
from dask.array.utils import asarray_safe
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from, random_state_data, typename
@derived_from(np.random.Generator, skipblocks=1)
def multivariate_hypergeometric(self, colors, nsample, size=None, method='marginals', chunks='auto', **kwargs):
    return _wrap_func(self, 'multivariate_hypergeometric', colors, nsample, size=size, method=method, chunks=chunks, **kwargs)