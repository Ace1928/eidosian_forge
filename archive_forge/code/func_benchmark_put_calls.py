import inspect
import logging
import numpy as np
import sys
from ray.util.client.ray_client_helpers import ray_start_client_server
from ray._private.ray_microbenchmark_helpers import timeit
def benchmark_put_calls(ray, results):

    def put_small():
        ray.put(0)
    results += timeit('client: put calls', put_small)