import types
import sys
import numbers
import functools
import copy
import inspect
def raise_with_traceback(exc, traceback=Ellipsis):
    if traceback == Ellipsis:
        _, _, traceback = sys.exc_info()
    raise exc.with_traceback(traceback)