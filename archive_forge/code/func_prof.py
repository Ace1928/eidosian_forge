import abc
import time
import warnings
from collections import namedtuple
from functools import wraps
from typing import Dict, Optional
def prof(fn=None, group: str='torchelastic'):
    """
    @profile decorator publishes duration.ms, count, success, failure metrics for the function that it decorates.

    The metric name defaults to the qualified name (``class_name.def_name``) of the function.
    If the function does not belong to a class, it uses the leaf module name instead.

    Usage

    ::

     @metrics.prof
     def x():
         pass

     @metrics.prof(group="agent")
     def y():
         pass
    """

    def wrap(f):

        @wraps(f)
        def wrapper(*args, **kwargs):
            key = _get_metric_name(f)
            try:
                start = time.time()
                result = f(*args, **kwargs)
                put_metric(f'{key}.success', 1, group)
            except Exception:
                put_metric(f'{key}.failure', 1, group)
                raise
            finally:
                put_metric(f'{key}.duration.ms', get_elapsed_time_ms(start), group)
            return result
        return wrapper
    if fn:
        return wrap(fn)
    else:
        return wrap