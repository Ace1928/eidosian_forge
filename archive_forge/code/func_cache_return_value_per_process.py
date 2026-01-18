import functools
import os
import subprocess
import sys
from mlflow.utils.os import is_windows
def cache_return_value_per_process(fn):
    """
    A decorator which globally caches the return value of the decorated function.
    But if current process forked out a new child process, in child process,
    old cache values are invalidated.

    Restrictions: The decorated function must be called with only positional arguments,
    and all the argument values must be hashable.
    """

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError('The function decorated by `cache_return_value_per_process` is not allowed to be called with key-word style arguments.')
        if (fn, args) in _per_process_value_cache_map:
            prev_value, prev_pid = _per_process_value_cache_map.get((fn, args))
            if os.getpid() == prev_pid:
                return prev_value
        new_value = fn(*args)
        new_pid = os.getpid()
        _per_process_value_cache_map[fn, args] = (new_value, new_pid)
        return new_value
    return wrapped_fn