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
def successful(self):
    """
        Returns true if none of the submitted tasks errored, else false. Should
        only be called once the result is ready (can be checked using `ready`).
        """
    if not self.ready():
        raise ValueError(f'{self!r} not ready')
    return not self._result_thread.got_error()