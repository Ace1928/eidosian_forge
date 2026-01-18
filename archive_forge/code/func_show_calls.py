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
def show_calls(show_args: bool=True, show_stack: bool=False, show_return: bool=False) -> Callable[..., Any]:
    """A method decorator to debug-log each call to the function."""

    def _decorator(func: AnyCallable) -> AnyCallable:

        @functools.wraps(func)
        def _wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            oid = getattr(self, OBJ_ID_ATTR, None)
            if oid is None:
                oid = f'{os.getpid():08d} {next(OBJ_IDS):04d}'
                setattr(self, OBJ_ID_ATTR, oid)
            extra = ''
            if show_args:
                eargs = ', '.join(map(repr, args))
                ekwargs = ', '.join(('{}={!r}'.format(*item) for item in kwargs.items()))
                extra += '('
                extra += eargs
                if eargs and ekwargs:
                    extra += ', '
                extra += ekwargs
                extra += ')'
            if show_stack:
                extra += ' @ '
                extra += '; '.join(short_stack(short_filenames=True).splitlines())
            callid = next(CALLS)
            msg = f'{oid} {callid:04d} {func.__name__}{extra}\n'
            DebugOutputFile.get_one(interim=True).write(msg)
            ret = func(self, *args, **kwargs)
            if show_return:
                msg = f'{oid} {callid:04d} {func.__name__} return {ret!r}\n'
                DebugOutputFile.get_one(interim=True).write(msg)
            return ret
        return _wrapper
    return _decorator