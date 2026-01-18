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
def apply_async(self, func: Callable, args: Optional[Tuple]=None, kwargs: Optional[Dict]=None, callback: Callable[[Any], None]=None, error_callback: Callable[[Exception], None]=None):
    """Run the given function on a random actor process and return an
        asynchronous interface to the result.

        Args:
            func: function to run.
            args: optional arguments to the function.
            kwargs: optional keyword arguments to the function.
            callback: callback to be executed on the result once it is finished
                only if it succeeds.
            error_callback: callback to be executed the result once it is
                finished only if the task errors. The exception raised by the
                task will be passed as the only argument to the callback.

        Returns:
            AsyncResult containing the result.
        """
    self._check_running()
    func = self._convert_to_ray_batched_calls_if_needed(func)
    object_ref = self._run_batch(self._next_actor_index(), func, [(args, kwargs)])
    return AsyncResult([object_ref], callback, error_callback, single_result=True)