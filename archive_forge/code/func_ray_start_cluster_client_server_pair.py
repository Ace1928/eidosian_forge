from contextlib import contextmanager
import time
from typing import Any, Dict
import ray as real_ray
from ray.job_config import JobConfig
import ray.util.client.server.server as ray_client_server
from ray.util.client import ray
from ray._private.client_mode_hook import enable_client_mode, disable_client_hook
@contextmanager
def ray_start_cluster_client_server_pair(address):
    ray._inside_client_test = True

    def ray_connect_handler(job_config=None, **ray_init_kwargs):
        real_ray.init(address=address)
    server = ray_client_server.serve('127.0.0.1:50051', ray_connect_handler=ray_connect_handler)
    ray.connect('127.0.0.1:50051')
    try:
        yield (ray, server)
    finally:
        ray._inside_client_test = False
        ray.disconnect()
        server.stop(0)