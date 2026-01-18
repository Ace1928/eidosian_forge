import importlib
import inspect
import os
import warnings
from eventlet import patcher
from eventlet.support import greenlets as greenlet
from eventlet import timeout
def get_hub():
    """Get the current event hub singleton object.

    .. note :: |internal|
    """
    try:
        hub = _threadlocal.hub
    except AttributeError:
        try:
            _threadlocal.Hub
        except AttributeError:
            use_hub()
        hub = _threadlocal.hub = _threadlocal.Hub()
    return hub