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
def bag_map(func, *args, **kwargs):
    """Apply a function elementwise across one or more bags.

    Note that all ``Bag`` arguments must be partitioned identically.

    Parameters
    ----------
    func : callable
    *args, **kwargs : Bag, Item, Delayed, or object
        Arguments and keyword arguments to pass to ``func``. Non-Bag args/kwargs
        are broadcasted across all calls to ``func``.

    Notes
    -----
    For calls with multiple `Bag` arguments, corresponding partitions should
    have the same length; if they do not, the call will error at compute time.

    Examples
    --------
    >>> import dask.bag as db
    >>> b = db.from_sequence(range(5), npartitions=2)
    >>> b2 = db.from_sequence(range(5, 10), npartitions=2)

    Apply a function to all elements in a bag:

    >>> db.map(lambda x: x + 1, b).compute()
    [1, 2, 3, 4, 5]

    Apply a function with arguments from multiple bags:

    >>> from operator import add
    >>> db.map(add, b, b2).compute()
    [5, 7, 9, 11, 13]

    Non-bag arguments are broadcast across all calls to the mapped function:

    >>> db.map(add, b, 1).compute()
    [1, 2, 3, 4, 5]

    Keyword arguments are also supported, and have the same semantics as
    regular arguments:

    >>> def myadd(x, y=0):
    ...     return x + y
    >>> db.map(myadd, b, y=b2).compute()
    [5, 7, 9, 11, 13]
    >>> db.map(myadd, b, y=1).compute()
    [1, 2, 3, 4, 5]

    Both arguments and keyword arguments can also be instances of
    ``dask.bag.Item`` or ``dask.delayed.Delayed``. Here we'll add the max value
    in the bag to each element:

    >>> db.map(myadd, b, b.max()).compute()
    [4, 5, 6, 7, 8]
    """
    name = '{}-{}'.format(funcname(func), tokenize(func, 'map', *args, **kwargs))
    dependencies = []
    bags = []
    args2 = []
    for a in args:
        if isinstance(a, Bag):
            bags.append(a)
            args2.append(a)
        elif isinstance(a, (Item, Delayed)):
            dependencies.append(a)
            args2.append((itertools.repeat, a.key))
        else:
            args2.append((itertools.repeat, a))
    bag_kwargs = {}
    other_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, Bag):
            bag_kwargs[k] = v
            bags.append(v)
        else:
            other_kwargs[k] = v
    other_kwargs, collections = unpack_scalar_dask_kwargs(other_kwargs)
    dependencies.extend(collections)
    if not bags:
        raise ValueError('At least one argument must be a Bag.')
    npartitions = {b.npartitions for b in bags}
    if len(npartitions) > 1:
        raise ValueError('All bags must have the same number of partitions.')
    npartitions = npartitions.pop()

    def build_iters(n):
        args = [(a.name, n) if isinstance(a, Bag) else a for a in args2]
        if bag_kwargs:
            args.extend(((b.name, n) for b in bag_kwargs.values()))
        return args
    if bag_kwargs:
        iter_kwarg_keys = list(bag_kwargs)
    else:
        iter_kwarg_keys = None
    dsk = {(name, n): (reify, (map_chunk, func, build_iters(n), iter_kwarg_keys, other_kwargs)) for n in range(npartitions)}
    return_type = set(map(type, bags))
    return_type = return_type.pop() if len(return_type) == 1 else Bag
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=bags + dependencies)
    return return_type(graph, name, npartitions)