from __future__ import annotations
import operator
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial, wraps
from numbers import Integral, Number
from operator import getitem
from pprint import pformat
from typing import Any, ClassVar, Literal, cast
import numpy as np
import pandas as pd
from pandas.api.types import (
from tlz import first, merge, partition_all, remove, unique
import dask.array as da
from dask import config, core
from dask.array.core import Array, normalize_arg
from dask.bag import map_partitions as map_bag_partitions
from dask.base import (
from dask.blockwise import Blockwise, BlockwiseDep, BlockwiseDepDict, blockwise
from dask.context import globalmethod
from dask.dataframe import methods
from dask.dataframe._compat import (
from dask.dataframe.accessor import CachedAccessor, DatetimeAccessor, StringAccessor
from dask.dataframe.categorical import CategoricalAccessor, categorize
from dask.dataframe.dispatch import (
from dask.dataframe.optimize import optimize
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameTreeReduction
from dask.typing import Graph, NestedKeys, no_default
from dask.utils import (
from dask.widgets import get_template
def repartition_divisions(a, b, name, out1, out2, force=False):
    """dask graph to repartition dataframe by new divisions

    Parameters
    ----------
    a : tuple
        old divisions
    b : tuple, listmypy
    out2 : str
        name of new dataframe
    force : bool, default False
        Allows the expansion of the existing divisions.
        If False then the new divisions lower and upper bounds must be
        the same as the old divisions.

    Examples
    --------
    >>> from pprint import pprint
    >>> pprint(repartition_divisions([1, 3, 7], [1, 4, 6, 7], 'a', 'b', 'c'))  # doctest: +ELLIPSIS
    {('b', 0): (<function boundary_slice at ...>, ('a', 0), 1, 3, False),
     ('b', 1): (<function boundary_slice at ...>, ('a', 1), 3, 4, False),
     ('b', 2): (<function boundary_slice at ...>, ('a', 1), 4, 6, False),
     ('b', 3): (<function boundary_slice at ...>, ('a', 1), 6, 7, True),
     ('c', 0): (<function concat at ...>, [('b', 0), ('b', 1)]),
     ('c', 1): ('b', 2),
     ('c', 2): ('b', 3)}
    """
    check_divisions(b)
    if len(b) < 2:
        raise ValueError('New division must be longer than 2 elements')
    if force:
        if a[0] < b[0]:
            msg = 'left side of the new division must be equal or smaller than old division'
            raise ValueError(msg)
        if a[-1] > b[-1]:
            msg = 'right side of the new division must be equal or larger than old division'
            raise ValueError(msg)
    else:
        if a[0] != b[0]:
            msg = 'left side of old and new divisions are different'
            raise ValueError(msg)
        if a[-1] != b[-1]:
            msg = 'right side of old and new divisions are different'
            raise ValueError(msg)

    def _is_single_last_div(x):
        """Whether last division only contains single label"""
        return len(x) >= 2 and x[-1] == x[-2]
    c = [a[0]]
    d = dict()
    low = a[0]
    i, j = (1, 1)
    k = 0
    last_elem = _is_single_last_div(a)
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            d[out1, k] = (methods.boundary_slice, (name, i - 1), low, a[i], False)
            low = a[i]
            i += 1
        elif a[i] > b[j]:
            d[out1, k] = (methods.boundary_slice, (name, i - 1), low, b[j], False)
            low = b[j]
            j += 1
        else:
            d[out1, k] = (methods.boundary_slice, (name, i - 1), low, b[j], False)
            low = b[j]
            if len(a) == i + 1 or a[i] < a[i + 1]:
                j += 1
            i += 1
        c.append(low)
        k += 1
    if a[-1] < b[-1] or b[-1] == b[-2]:
        for _j in range(j, len(b)):
            m = len(a) - 2
            d[out1, k] = (methods.boundary_slice, (name, m), low, b[_j], False)
            low = b[_j]
            c.append(low)
            k += 1
    else:
        if last_elem and i < len(a):
            d[out1, k] = (methods.boundary_slice, (name, i - 1), a[i], a[i], False)
            k += 1
        c.append(a[-1])
    d[out1, k - 1] = d[out1, k - 1][:-1] + (True,)
    i, j = (0, 1)
    last_elem = _is_single_last_div(c)
    while j < len(b):
        tmp = []
        while c[i] < b[j]:
            tmp.append((out1, i))
            i += 1
        while last_elem and c[i] == b[-1] and (b[-1] != b[-2] or j == len(b) - 1) and (i < k):
            tmp.append((out1, i))
            i += 1
        if len(tmp) == 0:
            d[out2, j - 1] = (methods.boundary_slice, (name, 0), a[0], a[0], False)
        elif len(tmp) == 1:
            d[out2, j - 1] = tmp[0]
        else:
            if not tmp:
                raise ValueError('check for duplicate partitions\nold:\n%s\n\nnew:\n%s\n\ncombined:\n%s' % (pformat(a), pformat(b), pformat(c)))
            d[out2, j - 1] = (methods.concat, tmp)
        j += 1
    return d