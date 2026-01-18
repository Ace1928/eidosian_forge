import inspect
import logging
import numpy as np
import sys
from ray.util.client.ray_client_helpers import ray_start_client_server
from ray._private.ray_microbenchmark_helpers import timeit
def benchmark_get_calls(ray, results):
    value = ray.put(0)

    def get_small():
        ray.get(value)
    results += timeit('client: get calls', get_small)