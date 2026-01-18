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
def handle_out(out, result):
    """Handle out parameters

    If out is a dask.DataFrame, dask.Series or dask.Scalar then
    this overwrites the contents of it with the result
    """
    if isinstance(out, tuple):
        if len(out) == 1:
            out = out[0]
        elif len(out) > 1:
            raise NotImplementedError('The out parameter is not fully supported')
        else:
            out = None
    if out is not None and out.__class__ != result.__class__:
        raise TypeError('Mismatched types between result and out parameter. out=%s, result=%s' % (str(type(out)), str(type(result))))
    if isinstance(out, DataFrame):
        if len(out.columns) != len(result.columns):
            raise ValueError('Mismatched columns count between result and out parameter. out=%s, result=%s' % (str(len(out.columns)), str(len(result.columns))))
    if isinstance(out, (Series, DataFrame, Scalar)):
        out._meta = result._meta
        out._name = result._name
        out.dask = result.dask
        if not isinstance(out, Scalar):
            out._divisions = result.divisions
        return result
    elif out is not None:
        msg = 'The out parameter is not fully supported. Received type %s, expected %s ' % (typename(type(out)), typename(type(result)))
        raise NotImplementedError(msg)
    else:
        return result