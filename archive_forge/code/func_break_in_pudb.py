from __future__ import annotations
import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread
from typing import (
from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable
def break_in_pudb(func: AnyCallable) -> AnyCallable:
    """A function decorator to stop in the debugger for each call."""

    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        import pudb
        sys.stdout = sys.__stdout__
        pudb.set_trace()
        return func(*args, **kwargs)
    return _wrapper