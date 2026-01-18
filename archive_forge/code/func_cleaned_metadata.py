from typing import Iterable, List, Optional
import ray
import ray.cloudpickle as cloudpickle
from ray.data._internal.execution.interfaces import PhysicalOperator, RefBundle
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.util import _warn_on_high_parallelism, call_with_retry
from ray.data.block import Block
from ray.data.context import DataContext
from ray.data.datasource.datasource import ReadTask
def cleaned_metadata(read_task: ReadTask):
    block_meta = read_task.get_metadata()
    task_size = len(cloudpickle.dumps(read_task))
    if block_meta.size_bytes is None or task_size > block_meta.size_bytes:
        if task_size > TASK_SIZE_WARN_THRESHOLD_BYTES:
            print(f'WARNING: the read task size ({task_size} bytes) is larger than the reported output size of the task ({block_meta.size_bytes} bytes). This may be a size reporting bug in the datasource being read from.')
        block_meta.size_bytes = task_size
    return block_meta