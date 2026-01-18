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
def make_function_table_key(key_type: bytes, job_id: JobID, key: Optional[bytes]):
    if key is None:
        return b':'.join([key_type, job_id.hex().encode()])
    else:
        return b':'.join([key_type, job_id.hex().encode(), key])