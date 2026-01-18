import logging
import os
import socket
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import exception_cause, skip_exceptions
from ray.types import ObjectRef
from ray.util.placement_group import PlacementGroup
def sort_workers_by_ip_and_gpu_id(self, _first_ip: Optional[str]=None):
    """Reorder the workers by their node ip and the lowest GPU id.

        This is useful for collocating workers on the same node.

        Example:
            Given workers with the following attributes:
                worker_0: ip=1, gpu_ids=[1]
                worker_1: ip=0, gpu_ids=[0]
                worker_2: ip=1, gpu_ids=[0]
                worker_3: ip=0, gpu_ids=[1]

            The function will perform the following steps:
                1. Group by node IP:
                    ip=0: worker_1, worker_3
                    ip=1: worker_0, worker_2

                2. Sort each group by GPU ID:
                    ip=0: worker_1 (gpu_id=0), worker_3 (gpu_id=1)
                    ip=1: worker_2 (gpu_id=0), worker_0 (gpu_id=1)

            Resulting in the order: [worker_1, worker_3, worker_2, worker_0]

        Args:
            _first_ip: The first IP to group by.
                Hack to avoid OOMs.
                This is just a temporary solution for Train loading entire checkpoints
                into memory by ensuring that the rank 0 worker is on the same node as
                trainable, thus allowing for lazy checkpoint transfer to be used.
                See https://github.com/ray-project/ray/issues/33073
                for more context.
                TODO remove this argument.
        """
    ip_to_workers = defaultdict(list)
    if _first_ip is not None:
        ip_to_workers[_first_ip] = []
    for worker in self.workers:
        ip_to_workers[worker.metadata.node_ip].append(worker)

    def get_lowest_gpu_id(worker) -> int:
        gpu_ids = worker.metadata.resource_ids.get('GPU', [])
        if not gpu_ids:
            return 0
        try:
            return min((int(gpu_id) for gpu_id in gpu_ids))
        except ValueError:
            return min(gpu_ids)
    for node_ip in ip_to_workers:
        ip_to_workers[node_ip].sort(key=get_lowest_gpu_id)
    sorted_workers = []
    for workers in ip_to_workers.values():
        sorted_workers.extend(workers)
    self.workers = sorted_workers