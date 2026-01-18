import asyncio
import logging
from ray._private.ray_microbenchmark_helpers import timeit
from ray._private.ray_client_microbenchmark import main as client_microbenchmark_main
import numpy as np
import multiprocessing
import ray
def async_actor():
    ray.get([a.small_value_with_arg.remote(i) for i in range(1000)])