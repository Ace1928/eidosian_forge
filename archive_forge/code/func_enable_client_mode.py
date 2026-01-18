import os
import threading
from contextlib import contextmanager
from functools import wraps
from ray._private.auto_init_hook import auto_init_ray
@contextmanager
def enable_client_mode():
    _explicitly_enable_client_mode()
    try:
        yield None
    finally:
        _explicitly_disable_client_mode()