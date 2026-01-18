import asyncio
import contextlib
import gc
import io
import sys
import traceback
import types
import typing
import unittest
import tornado
from tornado import web, gen, httpclient
from tornado.test.util import skipNotCPython
@contextlib.contextmanager
def assert_no_cycle_garbage():
    """Raise AssertionError if the wrapped code creates garbage with cycles."""
    gc.disable()
    gc.collect()
    gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_SAVEALL)
    yield
    try:
        f = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = f
        try:
            gc.collect()
        finally:
            sys.stderr = old_stderr
        garbage = gc.garbage[:]
        gc.garbage[:] = []
        if len(garbage) == 0:
            return
        for circular in find_circular_references(garbage):
            f.write('\n==========\n Circular \n==========')
            for item in circular:
                f.write(f'\n    {repr(item)}')
            for item in circular:
                if isinstance(item, types.FrameType):
                    f.write(f'\nLocals: {item.f_locals}')
                    f.write(f'\nTraceback: {repr(item)}')
                    traceback.print_stack(item)
        del garbage
        raise AssertionError(f.getvalue())
    finally:
        gc.set_debug(0)
        gc.enable()