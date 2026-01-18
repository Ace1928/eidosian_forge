from __future__ import annotations
import logging
import os
import time
import traceback
from collections.abc import Iterable
from glob import glob
from typing import TYPE_CHECKING, Any, ClassVar
import numpy as np
from xarray.conventions import cf_encoder
from xarray.core import indexing
from xarray.core.utils import FrozenDict, NdimSizeLenMixin, is_remote_uri
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def set_dimensions(self, variables, unlimited_dims=None):
    """
        This provides a centralized method to set the dimensions on the data
        store.

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
        """
    if unlimited_dims is None:
        unlimited_dims = set()
    existing_dims = self.get_dimensions()
    dims = {}
    for v in unlimited_dims:
        dims[v] = None
    for v in variables.values():
        dims.update(dict(zip(v.dims, v.shape)))
    for dim, length in dims.items():
        if dim in existing_dims and length != existing_dims[dim]:
            raise ValueError(f'Unable to update size for existing dimension{dim!r} ({length} != {existing_dims[dim]})')
        elif dim not in existing_dims:
            is_unlimited = dim in unlimited_dims
            self.set_dimension(dim, length, is_unlimited)