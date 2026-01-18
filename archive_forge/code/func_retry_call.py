import functools
import time
import inspect
import collections
import types
import itertools
import warnings
import setuptools.extern.more_itertools
from typing import Callable, TypeVar
def retry_call(func, cleanup=lambda: None, retries=0, trap=()):
    """
    Given a callable func, trap the indicated exceptions
    for up to 'retries' times, invoking cleanup on the
    exception. On the final attempt, allow any exceptions
    to propagate.
    """
    attempts = itertools.count() if retries == float('inf') else range(retries)
    for attempt in attempts:
        try:
            return func()
        except trap:
            cleanup()
    return func()