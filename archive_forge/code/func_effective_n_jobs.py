import logging
from typing import Any, Dict, Optional
from joblib import Parallel
from joblib._parallel_backends import MultiprocessingBackend
from joblib.pool import PicklingPool
import ray
from ray._private.usage import usage_lib
from ray.util.multiprocessing.pool import Pool
def effective_n_jobs(self, n_jobs):
    eff_n_jobs = super(RayBackend, self).effective_n_jobs(n_jobs)
    if n_jobs == -1:
        ray_cpus = int(ray._private.state.cluster_resources()['CPU'])
        eff_n_jobs = ray_cpus
    return eff_n_jobs