import ray
import os
from functools import wraps
import threading
def auto_init_ray():
    if enable_auto_connect and (not ray.is_initialized()):
        auto_init_lock.acquire()
        if not ray.is_initialized():
            ray.init()
        auto_init_lock.release()