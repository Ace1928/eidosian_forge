import collections
from types import GeneratorType
from typing import Any, Callable, Iterable, Iterator, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from ray.data._internal.compute import get_compute
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.execution.util import make_callable_class_concurrent
from ray.data._internal.logical.operators.map_operator import (
from ray.data._internal.numpy_support import is_valid_udf_return
from ray.data._internal.util import _truncated_repr
from ray.data.block import (
from ray.data.context import DataContext
def generate_filter_fn(target_max_block_size: int) -> Callable[[Iterator[Block], TaskContext, UserDefinedFunction], Iterator[Block]]:
    """Generate function to apply the UDF to each record of blocks,
    and filter out records that do not satisfy the given predicate.
    """
    context = DataContext.get_current()

    def fn(blocks: Iterator[Block], ctx: TaskContext, row_fn: UserDefinedFunction) -> Iterator[Block]:
        DataContext._set_current(context)
        transform_fn = _generate_transform_fn_for_filter(row_fn)
        map_transformer = _create_map_transformer_for_row_based_map_op(transform_fn)
        map_transformer.set_target_max_block_size(target_max_block_size)
        yield from map_transformer.apply_transform(blocks, ctx)
    return fn