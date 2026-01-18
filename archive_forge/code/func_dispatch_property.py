from __future__ import annotations
import operator
import numpy as np
import pytest
import dask.array as da
from dask.array import Array
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.utils import assert_eq
def dispatch_property(prop_name):
    """
    Wrap a simple property.
    """

    @property
    def wrapped(self, *a, **kw):
        return getattr(self.arr, prop_name)
    return wrapped