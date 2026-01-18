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
def incremental_resource_usage(self) -> ExecutionResources:
    if self._autoscaling_policy.should_scale_up(num_total_workers=self._actor_pool.num_total_actors(), num_running_workers=self._actor_pool.num_running_actors()):
        num_cpus = self._ray_remote_args.get('num_cpus', 0)
        num_gpus = self._ray_remote_args.get('num_gpus', 0)
    else:
        num_cpus = 0
        num_gpus = 0
    return ExecutionResources(cpu=num_cpus, gpu=num_gpus, object_store_memory=self._metrics.average_bytes_outputs_per_task)