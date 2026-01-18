import copy
import functools
import itertools
from typing import (
import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import (
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.rules.operator_fusion import _are_remote_args_compatible
from ray.data._internal.logical.rules.set_read_parallelism import (
from ray.data._internal.planner.plan_read_op import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import (
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.debug import log_once
def is_read_stage_equivalent(self) -> bool:
    """Return whether this plan can be executed as only a read stage."""
    from ray.data._internal.stage_impl import RandomizeBlocksStage
    context = self._context
    remaining_stages = self._stages_after_snapshot
    if context.optimize_fuse_stages and remaining_stages and isinstance(remaining_stages[0], RandomizeBlocksStage):
        remaining_stages = remaining_stages[1:]
    return self.has_lazy_input() and (not self._stages_before_snapshot) and (not remaining_stages) and (not self._snapshot_blocks or isinstance(self._snapshot_blocks, LazyBlockList))