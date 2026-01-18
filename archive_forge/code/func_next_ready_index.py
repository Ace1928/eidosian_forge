import collections
import copy
import gc
import itertools
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import ray
from ray._private.usage import usage_lib
from ray.util import log_once
def next_ready_index(self, timeout=None):
    try:
        return self._ready_index_queue.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError