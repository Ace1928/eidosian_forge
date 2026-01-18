import datetime
import time
from collections import deque
from contextlib import contextmanager
from weakref import proxy
from dateutil.parser import isoparse
from kombu.utils.objects import cached_property
from vine import Thenable, barrier, promise
from . import current_app, states
from ._state import _set_task_join_will_block, task_join_will_block
from .app import app_or_default
from .exceptions import ImproperlyConfigured, IncompleteStream, TimeoutError
from .utils.graph import DependencyGraph, GraphFormatter
def result_from_tuple(r, app=None):
    """Deserialize result from tuple."""
    app = app_or_default(app)
    Result = app.AsyncResult
    if not isinstance(r, ResultBase):
        res, nodes = r
        id, parent = res if isinstance(res, (list, tuple)) else (res, None)
        if parent:
            parent = result_from_tuple(parent, app)
        if nodes is not None:
            return app.GroupResult(id, [result_from_tuple(child, app) for child in nodes], parent=parent)
        return Result(id, parent=parent)
    return r