import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def execute_computation(thread_index: int):
    try:
        for item in fn(thread_safe_generator):
            if output_queue.put(item):
                return
        output_queue.put(Sentinel(thread_index))
    except Exception as e:
        output_queue.put(e)