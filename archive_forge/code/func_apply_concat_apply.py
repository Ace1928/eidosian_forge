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
@insert_meta_param_description
def apply_concat_apply(args, chunk=None, aggregate=None, combine=None, meta=no_default, token=None, chunk_kwargs=None, aggregate_kwargs=None, combine_kwargs=None, split_every=None, split_out=None, split_out_setup=None, split_out_setup_kwargs=None, sort=None, ignore_index=False, **kwargs):
    """Apply a function to blocks, then concat, then apply again

    Parameters
    ----------
    args :
        Positional arguments for the `chunk` function. All `dask.dataframe`
        objects should be partitioned and indexed equivalently.
    chunk : function [block-per-arg] -> block
        Function to operate on each block of data
    aggregate : function concatenated-block -> block
        Function to operate on the concatenated result of chunk
    combine : function concatenated-block -> block, optional
        Function to operate on intermediate concatenated results of chunk
        in a tree-reduction. If not provided, defaults to aggregate.
    $META
    token : str, optional
        The name to use for the output keys.
    chunk_kwargs : dict, optional
        Keywords for the chunk function only.
    aggregate_kwargs : dict, optional
        Keywords for the aggregate function only.
    combine_kwargs : dict, optional
        Keywords for the combine function only.
    split_every : int, optional
        Group partitions into groups of this size while performing a
        tree-reduction. If set to False, no tree-reduction will be used,
        and all intermediates will be concatenated and passed to ``aggregate``.
        Default is 8.
    split_out : int, optional
        Number of output partitions. Split occurs after first chunk reduction.
    split_out_setup : callable, optional
        If provided, this function is called on each chunk before performing
        the hash-split. It should return a pandas object, where each row
        (excluding the index) is hashed. If not provided, the chunk is hashed
        as is.
    split_out_setup_kwargs : dict, optional
        Keywords for the `split_out_setup` function only.
    sort : bool, default None
        If allowed, sort the keys of the output aggregation.
    ignore_index : bool, default False
        If True, do not preserve index values throughout ACA operations.
    kwargs :
        All remaining keywords will be passed to ``chunk``, ``aggregate``, and
        ``combine``.

    Examples
    --------
    >>> def chunk(a_block, b_block):
    ...     pass

    >>> def agg(df):
    ...     pass

    >>> apply_concat_apply([a, b], chunk=chunk, aggregate=agg)  # doctest: +SKIP
    """
    if split_out is None:
        split_out = 1
    if chunk_kwargs is None:
        chunk_kwargs = dict()
    if aggregate_kwargs is None:
        aggregate_kwargs = dict()
    chunk_kwargs.update(kwargs)
    aggregate_kwargs.update(kwargs)
    if combine is None:
        if combine_kwargs:
            raise ValueError('`combine_kwargs` provided with no `combine`')
        combine = aggregate
        combine_kwargs = aggregate_kwargs
    else:
        if combine_kwargs is None:
            combine_kwargs = dict()
        combine_kwargs.update(kwargs)
    if not isinstance(args, (tuple, list)):
        args = [args]
    dfs = [arg for arg in args if isinstance(arg, _Frame)]
    npartitions = {arg.npartitions for arg in dfs}
    if len(npartitions) > 1:
        raise ValueError('All arguments must have same number of partitions')
    npartitions = npartitions.pop()
    if split_every is None:
        split_every = 8
    elif split_every is False:
        split_every = npartitions
    elif split_every < 2 or not isinstance(split_every, Integral):
        raise ValueError('split_every must be an integer >= 2')
    token_key = tokenize(token or (chunk, aggregate), meta, args, chunk_kwargs, aggregate_kwargs, combine_kwargs, split_every, split_out, split_out_setup, split_out_setup_kwargs)
    chunk_name = f'{token or funcname(chunk)}-chunk-{token_key}'
    chunked = map_bag_partitions(chunk, *[arg.to_bag(format='frame') if isinstance(arg, _Frame) else arg for arg in args], token=chunk_name, **chunk_kwargs)
    if split_out and split_out > 1:
        chunked = chunked.map_partitions(hash_shard, split_out, split_out_setup, split_out_setup_kwargs, ignore_index, token='split-%s' % token_key)
    if sort is not None:
        if sort and split_out > 1:
            raise NotImplementedError('Cannot guarantee sorted keys for `split_out>1`. Try using split_out=1, or grouping with sort=False.')
        aggregate_kwargs = aggregate_kwargs or {}
        aggregate_kwargs['sort'] = sort
    final_name = f'{token or funcname(aggregate)}-agg-{token_key}'
    layer = DataFrameTreeReduction(final_name, chunked.name, npartitions, partial(_concat, ignore_index=ignore_index), partial(combine, **combine_kwargs) if combine_kwargs else combine, finalize_func=partial(aggregate, **aggregate_kwargs) if aggregate_kwargs else aggregate, split_every=split_every, split_out=split_out if split_out and split_out > 1 else None, tree_node_name=f'{token or funcname(combine)}-combine-{token_key}')
    if meta is no_default:
        meta_chunk = _emulate(chunk, *args, udf=True, **chunk_kwargs)
        meta = _emulate(aggregate, _concat([meta_chunk], ignore_index), udf=True, **aggregate_kwargs)
    meta = make_meta(meta, index=getattr(make_meta(dfs[0]), 'index', None) if dfs else None, parent_meta=dfs[0]._meta)
    graph = HighLevelGraph.from_collections(final_name, layer, dependencies=(chunked,))
    divisions = [None] * ((split_out or 1) + 1)
    return new_dd_object(graph, final_name, meta, divisions, parent_meta=dfs[0]._meta)