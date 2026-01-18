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
def lazify_task(task, start=True):
    """
    Given a task, remove unnecessary calls to ``list`` and ``reify``.

    This traverses tasks and small lists.  We choose not to traverse down lists
    of size >= 50 because it is unlikely that sequences this long contain other
    sequences in practice.

    Examples
    --------
    >>> def inc(x):
    ...     return x + 1
    >>> task = (sum, (list, (map, inc, [1, 2, 3])))
    >>> lazify_task(task)  # doctest: +ELLIPSIS
    (<built-in function sum>, (<class 'map'>, <function inc at ...>, [1, 2, 3]))
    """
    if type(task) is list and len(task) < 50:
        return [lazify_task(arg, False) for arg in task]
    if not istask(task):
        return task
    head, tail = (task[0], task[1:])
    if not start and head in (list, reify):
        task = task[1]
        return lazify_task(*tail, start=False)
    else:
        return (head,) + tuple((lazify_task(arg, False) for arg in tail))