import asyncio
import logging
from ray._private.ray_microbenchmark_helpers import timeit
from ray._private.ray_client_microbenchmark import main as client_microbenchmark_main
import numpy as np
import multiprocessing
import ray
@ray.remote
def do_put():
    for _ in range(10):
        ray.put(np.zeros(10 * 1024 * 1024, dtype=np.int64))