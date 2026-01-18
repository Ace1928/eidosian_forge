import dis
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from typing import Optional, Callable
import ray
import ray._private.profiling as profiling
from ray import cloudpickle as pickle
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.utils import (
from ray._private.serialization import pickle_dumps
from ray._raylet import (
def load_function_or_class_from_local(self, module_name, function_or_class_name):
    """Try to load a function or class in the module from local."""
    module = importlib.import_module(module_name)
    parts = [part for part in function_or_class_name.split('.') if part]
    object = module
    try:
        for part in parts:
            object = getattr(object, part)
        return object
    except Exception:
        return None