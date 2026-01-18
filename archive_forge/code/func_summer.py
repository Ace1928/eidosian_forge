from __future__ import annotations
import pytest
import operator
import numpy as np
import dask.array as da
from dask.array.chunk import coarsen, getitem, keepdims_wrapper
def summer(a, axis=None):
    return a.sum(axis=axis)