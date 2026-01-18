from __future__ import annotations
import io
import itertools
import math
import operator
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from functools import partial, reduce, wraps
from random import Random
from urllib.request import urlopen
import tlz as toolz
from fsspec.core import open_files
from tlz import (
from dask import config
from dask.bag import chunk
from dask.bag.avro import to_avro
from dask.base import (
from dask.blockwise import blockwise
from dask.context import globalmethod
from dask.core import flatten, get_dependencies, istask, quote, reverse_dict
from dask.delayed import Delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull, fuse, inline
from dask.sizeof import sizeof
from dask.typing import Graph, NestedKeys, no_default
from dask.utils import (
def map_chunk(f, iters, iter_kwarg_keys=None, kwargs=None):
    """Map ``f`` across one or more iterables, maybe with keyword arguments.

    Low-level function used in ``bag_map``, not user facing.

    Arguments
    ---------
    f : callable
    iters : List[Iterable]
    iter_kwarg_keys : List[str] or None
        Keyword names to use for pair with the tail end of ``iters``, allowing
        keyword arguments to be passed in from iterators.
    kwargs : dict or None
        Additional constant keyword arguments to use on every call to ``f``.
    """
    if kwargs:
        f = partial(f, **kwargs)
    iters = [iter(a) for a in iters]
    return _MapChunk(f, iters, kwarg_keys=iter_kwarg_keys)