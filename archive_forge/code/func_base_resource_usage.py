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
def base_resource_usage(self) -> ExecutionResources:
    min_workers = self._autoscaling_policy.min_workers
    return ExecutionResources(cpu=self._ray_remote_args.get('num_cpus', 0) * min_workers, gpu=self._ray_remote_args.get('num_gpus', 0) * min_workers)