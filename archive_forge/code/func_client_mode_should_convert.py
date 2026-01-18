import os
import threading
from contextlib import contextmanager
from functools import wraps
from ray._private.auto_init_hook import auto_init_ray
def client_mode_should_convert():
    """Determines if functions should be converted to client mode."""
    return (is_client_mode_enabled or is_client_mode_enabled_by_default) and _get_client_hook_status_on_thread()