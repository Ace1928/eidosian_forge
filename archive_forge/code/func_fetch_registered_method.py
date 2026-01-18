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
def fetch_registered_method(self, key: str, timeout: Optional[int]=None) -> Optional[ImportedFunctionInfo]:
    vals = self._worker.gcs_client.internal_kv_get(key, KV_NAMESPACE_FUNCTION_TABLE, timeout=timeout)
    if vals is None:
        return None
    else:
        vals = pickle.loads(vals)
        fields = ['job_id', 'function_id', 'function_name', 'function', 'module', 'max_calls']
        return ImportedFunctionInfo._make((vals.get(field) for field in fields))