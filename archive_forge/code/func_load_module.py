import os
import sys
import threading
from wandb_watchdog.utils import platform
from wandb_watchdog.utils.compat import Event
def load_module(module_name):
    """Imports a module given its name and returns a handle to it."""
    try:
        __import__(module_name)
    except ImportError:
        raise ImportError('No module named %s' % module_name)
    return sys.modules[module_name]