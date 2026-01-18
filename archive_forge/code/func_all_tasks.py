import concurrent.futures
import contextvars
import functools
import inspect
import itertools
import types
import warnings
import weakref
from types import GenericAlias
from . import base_tasks
from . import coroutines
from . import events
from . import exceptions
from . import futures
from .coroutines import _is_coroutine
def all_tasks(loop=None):
    """Return a set of all tasks for the loop."""
    if loop is None:
        loop = events.get_running_loop()
    i = 0
    while True:
        try:
            tasks = list(_all_tasks)
        except RuntimeError:
            i += 1
            if i >= 1000:
                raise
        else:
            break
    return {t for t in tasks if futures._get_loop(t) is loop and (not t.done())}