from contextlib import contextmanager
import time
from typing import Any, Dict
import ray as real_ray
from ray.job_config import JobConfig
import ray.util.client.server.server as ray_client_server
from ray.util.client import ray
from ray._private.client_mode_hook import enable_client_mode, disable_client_hook
def connect_handler(job_config: JobConfig=None, **ray_init_kwargs: Dict[str, Any]):
    import ray
    with disable_client_hook():
        if not ray.is_initialized():
            return ray.init(address, job_config=job_config, **ray_init_kwargs)