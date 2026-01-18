from __future__ import annotations
import contextlib
import math
import operator
import os
import pickle
import re
import sys
import traceback
import uuid
import warnings
from bisect import bisect
from collections.abc import (
from functools import partial, reduce, wraps
from itertools import product, zip_longest
from numbers import Integral, Number
from operator import add, mul
from threading import Lock
from typing import Any, TypeVar, Union, cast
import numpy as np
from numpy.typing import ArrayLike
from tlz import accumulate, concat, first, frequencies, groupby, partition
from tlz.curried import pluck
from dask import compute, config, core
from dask.array import chunk
from dask.array.chunk import getitem
from dask.array.chunk_types import is_valid_array_chunk, is_valid_chunk_type
from dask.array.dispatch import (  # noqa: F401
from dask.array.numpy_compat import _Recurser
from dask.array.slicing import replace_ellipsis, setitem_array, slice_array
from dask.base import (
from dask.blockwise import blockwise as core_blockwise
from dask.blockwise import broadcast_dimensions
from dask.context import globalmethod
from dask.core import quote
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.layers import ArraySliceDep, reshapelist
from dask.sizeof import sizeof
from dask.typing import Graph, Key, NestedKeys
from dask.utils import (
from dask.widgets import get_template
from dask.array.optimization import fuse_slice, optimize
from dask.array.blockwise import blockwise
from dask.array.utils import compute_meta, meta_from_array
def from_zarr(url, component=None, storage_options=None, chunks=None, name=None, inline_array=False, **kwargs):
    """Load array from the zarr storage format

    See https://zarr.readthedocs.io for details about the format.

    Parameters
    ----------
    url: Zarr Array or str or MutableMapping
        Location of the data. A URL can include a protocol specifier like s3://
        for remote data. Can also be any MutableMapping instance, which should
        be serializable if used in multiple processes.
    component: str or None
        If the location is a zarr group rather than an array, this is the
        subcomponent that should be loaded, something like ``'foo/bar'``.
    storage_options: dict
        Any additional parameters for the storage backend (ignored for local
        paths)
    chunks: tuple of ints or tuples of ints
        Passed to :func:`dask.array.from_array`, allows setting the chunks on
        initialisation, if the chunking scheme in the on-disc dataset is not
        optimal for the calculations to follow.
    name : str, optional
         An optional keyname for the array.  Defaults to hashing the input
    kwargs:
        Passed to :class:`zarr.core.Array`.
    inline_array : bool, default False
        Whether to inline the zarr Array in the values of the task graph.
        See :meth:`dask.array.from_array` for an explanation.

    See Also
    --------
    from_array
    """
    import zarr
    storage_options = storage_options or {}
    if isinstance(url, zarr.Array):
        z = url
    elif isinstance(url, (str, os.PathLike)):
        if isinstance(url, os.PathLike):
            url = os.fspath(url)
        if storage_options:
            store = zarr.storage.FSStore(url, **storage_options)
        else:
            store = url
        z = zarr.Array(store, read_only=True, path=component, **kwargs)
    else:
        z = zarr.Array(url, read_only=True, path=component, **kwargs)
    chunks = chunks if chunks is not None else z.chunks
    if name is None:
        name = 'from-zarr-' + tokenize(z, component, storage_options, chunks, **kwargs)
    return from_array(z, chunks, name=name, inline_array=inline_array)