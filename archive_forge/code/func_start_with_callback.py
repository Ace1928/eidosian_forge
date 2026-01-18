import atexit
import operator
import os
import sys
import threading
import time
import traceback as _traceback
import warnings
import subprocess
import functools
from more_itertools import always_iterable
def start_with_callback(self, func, args=None, kwargs=None):
    """Start 'func' in a new thread T, then start self (and return T)."""
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    args = (func,) + args

    def _callback(func, *a, **kw):
        self.wait(states.STARTED)
        func(*a, **kw)
    t = threading.Thread(target=_callback, args=args, kwargs=kwargs)
    t.name = 'Bus Callback ' + t.name
    t.start()
    self.start()
    return t