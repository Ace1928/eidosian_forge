from contextlib import contextmanager
import time
from typing import Any, Dict
import ray as real_ray
from ray.job_config import JobConfig
import ray.util.client.server.server as ray_client_server
from ray.util.client import ray
from ray._private.client_mode_hook import enable_client_mode, disable_client_hook
@contextmanager
def ray_start_client_server(metadata=None, ray_connect_handler=None, **kwargs):
    with ray_start_client_server_pair(metadata=metadata, ray_connect_handler=ray_connect_handler, **kwargs) as pair:
        client, server = pair
        yield client