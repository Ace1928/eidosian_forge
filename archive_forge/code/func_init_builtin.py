from _imp import (lock_held, acquire_lock, release_lock,
from importlib._bootstrap import _ERR_MSG, _exec, _load, _builtin_from_name
from importlib._bootstrap_external import SourcelessFileLoader
from importlib import machinery
from importlib import util
import importlib
import os
import sys
import tokenize
import types
import warnings
def init_builtin(name):
    """**DEPRECATED**

    Load and return a built-in module by name, or None is such module doesn't
    exist
    """
    try:
        return _builtin_from_name(name)
    except ImportError:
        return None