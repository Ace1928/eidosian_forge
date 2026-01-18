import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_operator import MapOperator, _map_task
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.execution.util import locality_string
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
@classmethod
def from_compute_strategy(cls, compute_strategy: ActorPoolStrategy):
    """Convert a legacy ActorPoolStrategy to an AutoscalingConfig."""
    assert isinstance(compute_strategy, ActorPoolStrategy)
    return cls(min_workers=compute_strategy.min_size, max_workers=compute_strategy.max_size, max_tasks_in_flight=compute_strategy.max_tasks_in_flight_per_actor or DEFAULT_MAX_TASKS_IN_FLIGHT, ready_to_total_workers_ratio=compute_strategy.ready_to_total_workers_ratio)