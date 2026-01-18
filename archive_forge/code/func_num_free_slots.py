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
def num_free_slots(self) -> int:
    """Return the number of free slots for task execution."""
    if not self._num_tasks_in_flight:
        return 0
    return sum((max(0, self._max_tasks_in_flight - num_tasks_in_flight) for num_tasks_in_flight in self._num_tasks_in_flight.values()))