import os
from dataclasses import dataclass
from typing import Optional, Set
from horovod.ray.runner import Coordinator
from horovod.ray.utils import detect_nics, nics_to_env_var
from horovod.runner.common.util import secret, timeout
import ray
from ray.train._internal.utils import update_env_vars
from ray.train._internal.worker_group import Worker, WorkerGroup
from ray.train.backend import Backend, BackendConfig
from ray.util import PublicAPI
@property
def start_timeout(self):
    return timeout.Timeout(self.timeout_s, message='Timed out waiting for {activity}. Please check connectivity between servers. You may need to increase the --start-timeout parameter if you have too many servers.')