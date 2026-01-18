import inspect
import logging
import numpy as np
import sys
from ray.util.client.ray_client_helpers import ray_start_client_server
from ray._private.ray_microbenchmark_helpers import timeit
def benchmark_remote_put_calls(ray, results):

    @ray.remote
    def do_put_small():
        for _ in range(100):
            ray.put(0)

    def put_multi_small():
        ray.get([do_put_small.remote() for _ in range(10)])
    results += timeit('client: tasks and put batch', put_multi_small, 1000)