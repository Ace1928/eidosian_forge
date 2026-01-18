import math
import threading
import time
from typing import Dict, List
import ray
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def purge_thread_run():
    while True:
        time.sleep(PURGE_INTERVAL)
        ray.get(self._self_handle.purge_expired_requests.remote())