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
def starmap(self, func, iterable, chunksize=None):
    """Same as `map`, but unpacks each element of the iterable as the
        arguments to func like: [func(*args) for args in iterable].
        """
    return self._map_async(func, iterable, chunksize=chunksize, unpack_args=True).get()