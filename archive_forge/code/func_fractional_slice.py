from __future__ import annotations
import functools
import math
import operator
from collections import defaultdict
from collections.abc import Callable
from itertools import product
from typing import Any
import tlz as toolz
from tlz.curried import map
from dask.base import tokenize
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise_token
from dask.core import flatten
from dask.highlevelgraph import Layer
from dask.utils import apply, cached_cumsum, concrete, insert
def fractional_slice(task, axes):
    """

    >>> fractional_slice(('x', 5.1), {0: 2})
    (<built-in function getitem>, ('x', 5), (slice(-2, None, None),))

    >>> fractional_slice(('x', 3, 5.1), {0: 2, 1: 3})
    (<built-in function getitem>, ('x', 3, 5), (slice(None, None, None), slice(-3, None, None)))

    >>> fractional_slice(('x', 2.9, 5.1), {0: 2, 1: 3})
    (<built-in function getitem>, ('x', 3, 5), (slice(0, 2, None), slice(-3, None, None)))
    """
    rounded = (task[0],) + tuple((int(round(i)) for i in task[1:]))
    index = []
    for i, (t, r) in enumerate(zip(task[1:], rounded[1:])):
        depth = axes.get(i, 0)
        if isinstance(depth, tuple):
            left_depth = depth[0]
            right_depth = depth[1]
        else:
            left_depth = depth
            right_depth = depth
        if t == r:
            index.append(slice(None, None, None))
        elif t < r and right_depth:
            index.append(slice(0, right_depth))
        elif t > r and left_depth:
            index.append(slice(-left_depth, None))
        else:
            index.append(slice(0, 0))
    index = tuple(index)
    if all((ind == slice(None, None, None) for ind in index)):
        return task
    else:
        return (operator.getitem, rounded, index)