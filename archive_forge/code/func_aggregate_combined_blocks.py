import collections
import heapq
import random
from typing import (
import numpy as np
from ray._private.utils import _get_pyarrow_version
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.data._internal.arrow_ops import transform_polars, transform_pyarrow
from ray.data._internal.numpy_support import (
from ray.data._internal.table_block import TableBlockAccessor, TableBlockBuilder
from ray.data._internal.util import _truncated_repr, find_partitions
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.row import TableRow
@staticmethod
def aggregate_combined_blocks(blocks: List[Block], key: Union[str, List[str]], aggs: Tuple['AggregateFn'], finalize: bool) -> Tuple[Block, BlockMetadata]:
    """Aggregate sorted, partially combined blocks with the same key range.

        This assumes blocks are already sorted by key in ascending order,
        so we can do merge sort to get all the rows with the same key.

        Args:
            blocks: A list of partially combined and sorted blocks.
            key: The column name of key or None for global aggregation.
            aggs: The aggregations to do.
            finalize: Whether to finalize the aggregation. This is used as an
                optimization for cases where we repeatedly combine partially
                aggregated groups.

        Returns:
            A block of [k, v_1, ..., v_n] columns and its metadata where k is
            the groupby key and v_i is the corresponding aggregation result for
            the ith given aggregation.
            If key is None then the k column is omitted.
        """
    stats = BlockExecStats.builder()
    keys = key if isinstance(key, list) else [key]
    key_fn = (lambda r: tuple(r[r._row.schema.names[:len(keys)]])) if key is not None else lambda r: (0,)
    iter = heapq.merge(*[ArrowBlockAccessor(block).iter_rows(public_row_format=False) for block in blocks], key=key_fn)
    next_row = None
    builder = ArrowBlockBuilder()
    while True:
        try:
            if next_row is None:
                next_row = next(iter)
            next_keys = key_fn(next_row)
            next_key_names = next_row._row.schema.names[:len(keys)] if key is not None else None

            def gen():
                nonlocal iter
                nonlocal next_row
                while key_fn(next_row) == next_keys:
                    yield next_row
                    try:
                        next_row = next(iter)
                    except StopIteration:
                        next_row = None
                        break
            first = True
            accumulators = [None] * len(aggs)
            resolved_agg_names = [None] * len(aggs)
            for r in gen():
                if first:
                    count = collections.defaultdict(int)
                    for i in range(len(aggs)):
                        name = aggs[i].name
                        if count[name] > 0:
                            name = ArrowBlockAccessor._munge_conflict(name, count[name])
                        count[name] += 1
                        resolved_agg_names[i] = name
                        accumulators[i] = r[name]
                    first = False
                else:
                    for i in range(len(aggs)):
                        accumulators[i] = aggs[i].merge(accumulators[i], r[resolved_agg_names[i]])
            row = {}
            if key is not None:
                for next_key, next_key_name in zip(next_keys, next_key_names):
                    row[next_key_name] = next_key
            for agg, agg_name, accumulator in zip(aggs, resolved_agg_names, accumulators):
                if finalize:
                    row[agg_name] = agg.finalize(accumulator)
                else:
                    row[agg_name] = accumulator
            builder.add(row)
        except StopIteration:
            break
    ret = builder.build()
    return (ret, ArrowBlockAccessor(ret).get_metadata(None, exec_stats=stats.build()))