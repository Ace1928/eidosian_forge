import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional
import torch
import torch.distributed as dist
import ray
from ray.train._internal.utils import get_address_and_port
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import Backend, BackendConfig
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME
from ray.util import PublicAPI
def on_shutdown(self, worker_group: WorkerGroup, backend_config: TorchConfig):
    worker_group.execute(_shutdown_torch, destroy_process_group=len(worker_group) > 1)