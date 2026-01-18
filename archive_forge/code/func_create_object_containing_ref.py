import asyncio
import logging
from ray._private.ray_microbenchmark_helpers import timeit
from ray._private.ray_client_microbenchmark import main as client_microbenchmark_main
import numpy as np
import multiprocessing
import ray
@ray.remote
def create_object_containing_ref():
    obj_refs = []
    for _ in range(10000):
        obj_refs.append(ray.put(1))
    return obj_refs