from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import ray
from .ref_bundle import RefBundle
from ray._raylet import ObjectRefGenerator
from ray.data._internal.execution.interfaces.execution_options import (
from ray.data._internal.execution.interfaces.op_runtime_metrics import OpRuntimeMetrics
from ray.data._internal.logical.interfaces import Operator
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
def on_data_ready(self, max_blocks_to_read: Optional[int]) -> int:
    """Callback when data is ready to be read from the streaming generator.

        Args:
            max_blocks_to_read: Max number of blocks to read. If None, all available
                will be read.
        Returns: The number of blocks read.
        """
    num_blocks_read = 0
    while max_blocks_to_read is None or num_blocks_read < max_blocks_to_read:
        try:
            block_ref = self._streaming_gen._next_sync(0)
            if block_ref.is_nil():
                break
        except StopIteration:
            self._task_done_callback(None)
            break
        try:
            meta = ray.get(next(self._streaming_gen))
        except StopIteration:
            try:
                ray.get(block_ref)
                assert False, 'Above ray.get should raise an exception.'
            except Exception as ex:
                self._task_done_callback(ex)
                raise ex from None
        self._output_ready_callback(RefBundle([(block_ref, meta)], owns_blocks=True))
        num_blocks_read += 1
    return num_blocks_read