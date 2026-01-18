from __future__ import annotations
from collections.abc import Iterable
from functools import partial
from math import ceil
from operator import getitem
from threading import Lock
from typing import TYPE_CHECKING, Literal, overload
import numpy as np
import pandas as pd
import dask.array as da
from dask.base import is_dask_collection, tokenize
from dask.blockwise import BlockwiseDepDict, blockwise
from dask.dataframe._compat import is_any_real_numeric_dtype
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import (
from dask.dataframe.dispatch import meta_lib_from_array
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.dataframe.utils import (
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import M, funcname, is_arraylike
@insert_meta_param_description
def from_map(func, *iterables, args=None, meta=None, divisions=None, label=None, token=None, enforce_metadata=True, **kwargs):
    """Create a DataFrame collection from a custom function map

    WARNING: The ``from_map`` API is experimental, and stability is not
    yet guaranteed. Use at your own risk!

    Parameters
    ----------
    func : callable
        Function used to create each partition. If ``func`` satisfies the
        ``DataFrameIOFunction`` protocol, column projection will be enabled.
    *iterables : Iterable objects
        Iterable objects to map to each output partition. All iterables must
        be the same length. This length determines the number of partitions
        in the output collection (only one element of each iterable will
        be passed to ``func`` for each partition).
    args : list or tuple, optional
        Positional arguments to broadcast to each output partition. Note
        that these arguments will always be passed to ``func`` after the
        ``iterables`` positional arguments.
    $META
    divisions : tuple, str, optional
        Partition boundaries along the index.
        For tuple, see https://docs.dask.org/en/latest/dataframe-design.html#partitions
        For string 'sorted' will compute the delayed values to find index
        values.  Assumes that the indexes are mutually sorted.
        If None, then won't use index information
    label : str, optional
        String to use as the function-name label in the output
        collection-key names.
    token : str, optional
        String to use as the "token" in the output collection-key names.
    enforce_metadata : bool, default True
        Whether to enforce at runtime that the structure of the DataFrame
        produced by ``func`` actually matches the structure of ``meta``.
        This will rename and reorder columns for each partition,
        and will raise an error if this doesn't work,
        but it won't raise if dtypes don't match.
    **kwargs:
        Key-word arguments to broadcast to each output partition. These
        same arguments will be passed to ``func`` for every output partition.

    Examples
    --------
    >>> import pandas as pd
    >>> import dask.dataframe as dd
    >>> func = lambda x, size=0: pd.Series([x] * size)
    >>> inputs = ["A", "B"]
    >>> dd.from_map(func, inputs, size=2).compute()
    0    A
    1    A
    0    B
    1    B
    dtype: object

    This API can also be used as an alternative to other file-based
    IO functions, like ``read_parquet`` (which are already just
    ``from_map`` wrapper functions):

    >>> import pandas as pd
    >>> import dask.dataframe as dd
    >>> paths = ["0.parquet", "1.parquet", "2.parquet"]
    >>> dd.from_map(pd.read_parquet, paths).head()  # doctest: +SKIP
                        name
    timestamp
    2000-01-01 00:00:00   Laura
    2000-01-01 00:00:01  Oliver
    2000-01-01 00:00:02   Alice
    2000-01-01 00:00:03  Victor
    2000-01-01 00:00:04     Bob

    Since ``from_map`` allows you to map an arbitrary function
    to any number of iterable objects, it can be a very convenient
    means of implementing functionality that may be missing from
    from other DataFrame-creation methods. For example, if you
    happen to have apriori knowledge about the number of rows
    in each of the files in a dataset, you can generate a
    DataFrame collection with a global RangeIndex:

    >>> import pandas as pd
    >>> import numpy as np
    >>> import dask.dataframe as dd
    >>> paths = ["0.parquet", "1.parquet", "2.parquet"]
    >>> file_sizes = [86400, 86400, 86400]
    >>> def func(path, row_offset):
    ...     # Read parquet file and set RangeIndex offset
    ...     df = pd.read_parquet(path)
    ...     return df.set_index(
    ...         pd.RangeIndex(row_offset, row_offset+len(df))
    ...     )
    >>> def get_ddf(paths, file_sizes):
    ...     offsets = [0] + list(np.cumsum(file_sizes))
    ...     return dd.from_map(
    ...         func, paths, offsets[:-1], divisions=offsets
    ...     )
    >>> ddf = get_ddf(paths, file_sizes)  # doctest: +SKIP
    >>> ddf.index  # doctest: +SKIP
    Dask Index Structure:
    npartitions=3
    0         int64
    86400       ...
    172800      ...
    259200      ...
    dtype: int64
    Dask Name: myfunc, 6 tasks

    See Also
    --------
    dask.dataframe.from_delayed
    dask.layers.DataFrameIOLayer
    """
    if not callable(func):
        raise ValueError('`func` argument must be `callable`')
    lengths = set()
    iterables = list(iterables)
    for i, iterable in enumerate(iterables):
        if not isinstance(iterable, Iterable):
            raise ValueError(f'All elements of `iterables` must be Iterable, got {type(iterable)}')
        try:
            lengths.add(len(iterable))
        except (AttributeError, TypeError):
            iterables[i] = list(iterable)
            lengths.add(len(iterables[i]))
    if len(lengths) == 0:
        raise ValueError('`from_map` requires at least one Iterable input')
    elif len(lengths) > 1:
        raise ValueError('All `iterables` must have the same length')
    if lengths == {0}:
        raise ValueError('All `iterables` must have a non-zero length')
    produces_tasks = kwargs.pop('produces_tasks', False)
    creation_info = kwargs.pop('creation_info', None)
    if produces_tasks or len(iterables) == 1:
        if len(iterables) > 1:
            raise ValueError('Multiple iterables not supported when produces_tasks=True')
        inputs = iterables[0]
        packed = False
    else:
        inputs = list(zip(*iterables))
        packed = True
    label = label or funcname(func)
    token = token or tokenize(func, meta, inputs, args, divisions, enforce_metadata, **kwargs)
    name = f'{label}-{token}'
    column_projection = func.columns if isinstance(func, DataFrameIOFunction) else None
    if meta is None:
        meta = _emulate(func, *(inputs[0] if packed else inputs[:1]), *(args or []), udf=True, **kwargs)
        meta_is_emulated = True
    else:
        meta = make_meta(meta)
        meta_is_emulated = False
    if not (has_parallel_type(meta) or (is_arraylike(meta) and meta.shape)):
        if not meta_is_emulated:
            raise TypeError('Meta is not valid, `from_map` expects output to be a pandas object. Try passing a pandas object as meta or a dict or tuple representing the (name, dtype) of the columns.')
        meta = make_meta(_concat([meta]))
    meta = make_meta(meta)
    if packed or args or kwargs or enforce_metadata:
        io_func = _PackedArgCallable(func, args=args, kwargs=kwargs, meta=meta if enforce_metadata else None, enforce_metadata=enforce_metadata, packed=packed)
    else:
        io_func = func
    layer = DataFrameIOLayer(name, column_projection, inputs, io_func, label=label, produces_tasks=produces_tasks, creation_info=creation_info)
    divisions = divisions or [None] * (len(inputs) + 1)
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[])
    return new_dd_object(graph, name, meta, divisions)