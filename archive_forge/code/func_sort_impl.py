from typing import TYPE_CHECKING, List, Optional, Tuple, TypeVar, Union
import numpy as np
from ray.data._internal.block_list import BlockList
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.push_based_shuffle import PushBasedShufflePlan
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.shuffle import ShuffleOp, SimpleShufflePlan
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
def sort_impl(blocks: BlockList, clear_input_blocks: bool, sort_key: SortKey, ctx: Optional[TaskContext]=None) -> Tuple[BlockList, dict]:
    stage_info = {}
    blocks_list = blocks.get_blocks()
    if len(blocks_list) == 0:
        return (BlockList([], []), stage_info)
    num_mappers = len(blocks_list)
    num_reducers = num_mappers
    boundaries = sample_boundaries(blocks_list, sort_key, num_reducers, ctx)
    _, ascending = sort_key.to_pandas_sort_args()
    if not ascending:
        boundaries.reverse()
    context = DataContext.get_current()
    if context.use_push_based_shuffle:
        sort_op_cls = PushBasedSortOp
    else:
        sort_op_cls = SimpleSortOp
    sort_op = sort_op_cls(map_args=[boundaries, sort_key], reduce_args=[sort_key])
    return sort_op.execute(blocks, num_reducers, clear_input_blocks)