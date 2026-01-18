import logging
import random
import sys
import time
import traceback
from google.cloud.ml.util import _exceptions
from six import reraise
def real_decorator(fun):
    """The real decorator whose purpose is to return the wrapped function."""
    retry_intervals = iter(FuzzedExponentialIntervals(initial_delay_secs, num_retries, fuzz=0.5 if fuzz else 0))

    def wrapper(*args, **kwargs):
        while True:
            try:
                return fun(*args, **kwargs)
            except Exception as exn:
                if not retry_filter(exn):
                    raise
                exn_traceback = sys.exc_info()[2]
                try:
                    try:
                        sleep_interval = next(retry_intervals)
                    except StopIteration:
                        reraise(type(exn), exn, exn_traceback)
                    logger('Retry with exponential backoff: waiting for %s seconds before retrying %s because we caught exception: %s Traceback for above exception (most recent call last):\n%s', sleep_interval, getattr(fun, '__name__', str(fun)), ''.join(traceback.format_exception_only(exn.__class__, exn)), ''.join(traceback.format_tb(exn_traceback)))
                    clock.sleep(sleep_interval)
                finally:
                    if sys.version_info < (3, 0):
                        sys.exc_clear()
                    exn_traceback = None
    return wrapper