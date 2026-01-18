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
@insert_meta_param_description(pad=12)
def map_partitions(self, func, *args, **kwargs):
    """Apply Python function on each DataFrame partition.

        Note that the index and divisions are assumed to remain unchanged.

        Parameters
        ----------
        func : function
            The function applied to each partition. If this function accepts
            the special ``partition_info`` keyword argument, it will receive
            information on the partition's relative location within the
            dataframe.
        args, kwargs :
            Positional and keyword arguments to pass to the function.
            Positional arguments are computed on a per-partition basis, while
            keyword arguments are shared across all partitions. The partition
            itself will be the first positional argument, with all other
            arguments passed *after*. Arguments can be ``Scalar``, ``Delayed``,
            or regular Python objects. DataFrame-like args (both dask and
            pandas) will be repartitioned to align (if necessary) before
            applying the function; see ``align_dataframes`` to control this
            behavior.
        enforce_metadata : bool, default True
            Whether to enforce at runtime that the structure of the DataFrame
            produced by ``func`` actually matches the structure of ``meta``.
            This will rename and reorder columns for each partition,
            and will raise an error if this doesn't work,
            but it won't raise if dtypes don't match.
        transform_divisions : bool, default True
            Whether to apply the function onto the divisions and apply those
            transformed divisions to the output.
        align_dataframes : bool, default True
            Whether to repartition DataFrame- or Series-like args
            (both dask and pandas) so their divisions align before applying
            the function. This requires all inputs to have known divisions.
            Single-partition inputs will be split into multiple partitions.

            If False, all inputs must have either the same number of partitions
            or a single partition. Single-partition inputs will be broadcast to
            every partition of multi-partition inputs.
        $META

        Examples
        --------
        Given a DataFrame, Series, or Index, such as:

        >>> import pandas as pd
        >>> import dask.dataframe as dd
        >>> df = pd.DataFrame({'x': [1, 2, 3, 4, 5],
        ...                    'y': [1., 2., 3., 4., 5.]})
        >>> ddf = dd.from_pandas(df, npartitions=2)

        One can use ``map_partitions`` to apply a function on each partition.
        Extra arguments and keywords can optionally be provided, and will be
        passed to the function after the partition.

        Here we apply a function with arguments and keywords to a DataFrame,
        resulting in a Series:

        >>> def myadd(df, a, b=1):
        ...     return df.x + df.y + a + b
        >>> res = ddf.map_partitions(myadd, 1, b=2)
        >>> res.dtype
        dtype('float64')

        Here we apply a function to a Series resulting in a Series:

        >>> res = ddf.x.map_partitions(lambda x: len(x)) # ddf.x is a Dask Series Structure
        >>> res.dtype
        dtype('int64')

        By default, dask tries to infer the output metadata by running your
        provided function on some fake data. This works well in many cases, but
        can sometimes be expensive, or even fail. To avoid this, you can
        manually specify the output metadata with the ``meta`` keyword. This
        can be specified in many forms, for more information see
        ``dask.dataframe.utils.make_meta``.

        Here we specify the output is a Series with no name, and dtype
        ``float64``:

        >>> res = ddf.map_partitions(myadd, 1, b=2, meta=(None, 'f8'))

        Here we map a function that takes in a DataFrame, and returns a
        DataFrame with a new column:

        >>> res = ddf.map_partitions(lambda df: df.assign(z=df.x * df.y))
        >>> res.dtypes
        x      int64
        y    float64
        z    float64
        dtype: object

        As before, the output metadata can also be specified manually. This
        time we pass in a ``dict``, as the output is a DataFrame:

        >>> res = ddf.map_partitions(lambda df: df.assign(z=df.x * df.y),
        ...                          meta={'x': 'i8', 'y': 'f8', 'z': 'f8'})

        In the case where the metadata doesn't change, you can also pass in
        the object itself directly:

        >>> res = ddf.map_partitions(lambda df: df.head(), meta=ddf)

        Also note that the index and divisions are assumed to remain unchanged.
        If the function you're mapping changes the index/divisions, you'll need
        to clear them afterwards:

        >>> ddf.map_partitions(func).clear_divisions()  # doctest: +SKIP

        Your map function gets information about where it is in the dataframe by
        accepting a special ``partition_info`` keyword argument.

        >>> def func(partition, partition_info=None):
        ...     pass

        This will receive the following information:

        >>> partition_info  # doctest: +SKIP
        {'number': 1, 'division': 3}

        For each argument and keyword arguments that are dask dataframes you will
        receive the number (n) which represents the nth partition of the dataframe
        and the division (the first index value in the partition). If divisions
        are not known (for instance if the index is not sorted) then you will get
        None as the division.
        """
    return map_partitions(func, self, *args, **kwargs)