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
def foldby(self, key, binop, initial=no_default, combine=None, combine_initial=no_default, split_every=None):
    """Combined reduction and groupby.

        Foldby provides a combined groupby and reduce for efficient parallel
        split-apply-combine tasks.

        The computation

        >>> b.foldby(key, binop, init)                        # doctest: +SKIP

        is equivalent to the following:

        >>> def reduction(group):                               # doctest: +SKIP
        ...     return reduce(binop, group, init)               # doctest: +SKIP

        >>> b.groupby(key).map(lambda (k, v): (k, reduction(v)))# doctest: +SKIP

        But uses minimal communication and so is *much* faster.

        >>> import dask.bag as db
        >>> b = db.from_sequence(range(10))
        >>> iseven = lambda x: x % 2 == 0
        >>> add = lambda x, y: x + y
        >>> dict(b.foldby(iseven, add))
        {True: 20, False: 25}

        **Key Function**

        The key function determines how to group the elements in your bag.
        In the common case where your bag holds dictionaries then the key
        function often gets out one of those elements.

        >>> def key(x):
        ...     return x['name']

        This case is so common that it is special cased, and if you provide a
        key that is not a callable function then dask.bag will turn it into one
        automatically.  The following are equivalent:

        >>> b.foldby(lambda x: x['name'], ...)  # doctest: +SKIP
        >>> b.foldby('name', ...)  # doctest: +SKIP

        **Binops**

        It can be tricky to construct the right binary operators to perform
        analytic queries.  The ``foldby`` method accepts two binary operators,
        ``binop`` and ``combine``.  Binary operators two inputs and output must
        have the same type.

        Binop takes a running total and a new element and produces a new total:

        >>> def binop(total, x):
        ...     return total + x['amount']

        Combine takes two totals and combines them:

        >>> def combine(total1, total2):
        ...     return total1 + total2

        Each of these binary operators may have a default first value for
        total, before any other value is seen.  For addition binary operators
        like above this is often ``0`` or the identity element for your
        operation.

        **split_every**

        Group partitions into groups of this size while performing reduction.
        Defaults to 8.

        >>> b.foldby('name', binop, 0, combine, 0)  # doctest: +SKIP

        Examples
        --------

        We can compute the maximum of some ``(key, value)`` pairs, grouped
        by the ``key``. (You might be better off converting the ``Bag`` to
        a ``dask.dataframe`` and using its groupby).

        >>> import random
        >>> import dask.bag as db

        >>> tokens = list('abcdefg')
        >>> values = range(10000)
        >>> a = [(random.choice(tokens), random.choice(values))
        ...       for _ in range(100)]
        >>> a[:2]  # doctest: +SKIP
        [('g', 676), ('a', 871)]

        >>> a = db.from_sequence(a)

        >>> def binop(t, x):
        ...     return max((t, x), key=lambda x: x[1])

        >>> a.foldby(lambda x: x[0], binop).compute()  # doctest: +SKIP
        [('g', ('g', 984)),
         ('a', ('a', 871)),
         ('b', ('b', 999)),
         ('c', ('c', 765)),
         ('f', ('f', 955)),
         ('e', ('e', 991)),
         ('d', ('d', 854))]

        See Also
        --------

        toolz.reduceby
        pyspark.combineByKey
        """
    if split_every is None:
        split_every = 8
    if split_every is False:
        split_every = self.npartitions
    token = tokenize(self, key, binop, initial, combine, combine_initial)
    a = 'foldby-a-' + token
    if combine is None:
        combine = binop
    if initial is not no_default:
        dsk = {(a, i): (reduceby, key, binop, (self.name, i), initial) for i in range(self.npartitions)}
    else:
        dsk = {(a, i): (reduceby, key, binop, (self.name, i)) for i in range(self.npartitions)}
    combine2 = partial(chunk.foldby_combine2, combine)
    depth = 0
    k = self.npartitions
    b = a
    while k > split_every:
        c = b + str(depth)
        if combine_initial is not no_default:
            for i, inds in enumerate(partition_all(split_every, range(k))):
                dsk[c, i] = (reduceby, 0, combine2, (toolz.concat, (map, dictitems, [(b, j) for j in inds])), combine_initial)
        else:
            for i, inds in enumerate(partition_all(split_every, range(k))):
                dsk[c, i] = (merge_with, (partial, reduce, combine), [(b, j) for j in inds])
        k = i + 1
        b = c
        depth += 1
    e = 'foldby-b-' + token
    if combine_initial is not no_default:
        dsk[e, 0] = (dictitems, (reduceby, 0, combine2, (toolz.concat, (map, dictitems, [(b, j) for j in range(k)])), combine_initial))
    else:
        dsk[e, 0] = (dictitems, (merge_with, (partial, reduce, combine), [(b, j) for j in range(k)]))
    graph = HighLevelGraph.from_collections(e, dsk, dependencies=[self])
    return type(self)(graph, e, 1)