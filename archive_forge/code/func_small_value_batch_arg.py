import asyncio
import logging
from ray._private.ray_microbenchmark_helpers import timeit
from ray._private.ray_client_microbenchmark import main as client_microbenchmark_main
import numpy as np
import multiprocessing
import ray
def small_value_batch_arg(self, n):
    x = ray.put(0)
    results = []
    for s in self.servers:
        results.extend([s.small_value_arg.remote(x) for _ in range(n)])
    ray.get(results)