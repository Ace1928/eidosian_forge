import ray
import os
from functools import wraps
import threading
@wraps(fn)
def auto_init_wrapper(*args, **kwargs):
    auto_init_ray()
    return fn(*args, **kwargs)