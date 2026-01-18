import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
import ray
import ray._private.ray_constants as ray_constants
from ray._private.ray_constants import env_integer
from ray.data import Dataset
from ray.exceptions import RayActorError
from ray.train import Checkpoint, DataConfig
from ray.train._internal.session import (
from ray.train._internal.storage import StorageContext
from ray.train._internal.utils import check_for_failure
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import BackendConfig
from ray.train.constants import (
from ray.util.placement_group import get_current_placement_group, remove_placement_group
def start_training(self, train_func: Callable[[], T], datasets: Dict[str, Dataset], metadata: Dict[str, Any], data_config: DataConfig, storage: StorageContext, checkpoint: Optional[Checkpoint]=None, on_session_init: Callable[[], None]=None) -> None:
    """Executes a training function on all workers in a separate thread.

        ``finish_training`` should be called after this.

        Args:
            train_func: The training function to run on each worker.
            datasets: The base datasets.
            data_config: The config object for creating dataset shards for workers.
            checkpoint: The checkpoint data that
                should be loaded onto each worker and accessed by the
                training function via ``session.get_checkpoint()``. If this
                is ``None`` then no checkpoint will be loaded.
        """
    use_detailed_autofilled_metrics = env_integer(ENABLE_DETAILED_AUTOFILLED_METRICS_ENV, 0)

    def initialize_session(train_func, world_rank, local_rank, node_rank, local_world_size, world_size, trial_info, checkpoint, dataset_shard, metadata, storage):
        try:
            init_session(training_func=train_func, world_rank=world_rank, local_rank=local_rank, node_rank=node_rank, local_world_size=local_world_size, world_size=world_size, trial_info=trial_info, dataset_shard=dataset_shard, metadata=metadata, checkpoint=checkpoint, detailed_autofilled_metrics=use_detailed_autofilled_metrics, storage=storage)
        except ValueError:
            raise TrainBackendError('Attempting to start training but a previous training run is still ongoing. You must call `finish_training` before calling `start_training` again.')
    if self.dataset_shards is None:
        actors = [worker.actor for worker in self.worker_group.workers]
        node_ids = [worker.metadata.node_id for worker in self.worker_group.workers]
        self.dataset_shards = data_config.configure(datasets, world_size=len(self.worker_group), worker_handles=actors, worker_node_ids=node_ids)
    local_rank_map, local_world_size_map, node_rank_map = self._create_rank_world_size_mappings()
    futures = []
    for index in range(len(self.worker_group)):
        futures.append(self.worker_group.execute_single_async(index, initialize_session, world_rank=index, local_rank=local_rank_map[index], node_rank=node_rank_map[index], local_world_size=local_world_size_map[index], world_size=len(self.worker_group), trial_info=self._trial_info, train_func=train_func, dataset_shard=self.dataset_shards[index], metadata=metadata, checkpoint=checkpoint, storage=storage))
    self._backend.on_training_start(self.worker_group, self._backend_config)
    self.get_with_failure_handling(futures)
    if on_session_init:
        on_session_init()

    def train_async():
        session = get_session()
        session.start()
    self.worker_group.execute_async(train_async)