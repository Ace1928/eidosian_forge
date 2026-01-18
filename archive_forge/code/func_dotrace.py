import logging
import sys
import types
import threading
import inspect
from functools import wraps
from itertools import chain
from numba.core import config
def dotrace(*args, **kwds):
    """Function decorator to trace a function's entry and exit.

    *args: categories in which to trace this function. Example usage:

    @trace
    def function(...):...

    @trace('mycategory')
    def function(...):...


    """
    recursive = kwds.get('recursive', False)

    def decorator(func):
        spec = None
        logger = logging.getLogger('trace')

        def wrapper(*args, **kwds):
            if not logger.isEnabledFor(logging.INFO) or tls.tracing:
                return func(*args, **kwds)
            fname, ftype = find_function_info(func, spec, args)
            try:
                tls.tracing = True
                enter, leave = create_events(fname, spec, args, kwds)
                try:
                    logger.info(''.join(enter))
                    tls.indent += 1
                    try:
                        try:
                            tls.tracing = False
                            result = func(*args, **kwds)
                        finally:
                            tls.tracing = True
                    except:
                        type, value, traceback = sys.exc_info()
                        leave.append(' => exception thrown\n\traise ')
                        mname = type.__module__
                        if mname != '__main__':
                            leave.append(mname)
                            leave.append('.')
                        leave.append(type.__name__)
                        if value.args:
                            leave.append('(')
                            leave.append(', '.join((chop(v) for v in value.args)))
                            leave.append(')')
                        else:
                            leave.append('()')
                        raise
                    else:
                        if result is not None:
                            leave.append(' -> ')
                            leave.append(chop(result))
                finally:
                    tls.indent -= 1
                    logger.info(''.join(leave))
            finally:
                tls.tracing = False
            return result
        result = None
        rewrap = lambda x: x
        if type(func) == classmethod:
            rewrap = type(func)
            func = func.__get__(True).__func__
        elif type(func) == staticmethod:
            rewrap = type(func)
            func = func.__get__(True)
        elif type(func) == property:
            raise NotImplementedError
        spec = inspect.getfullargspec(func)
        return rewrap(wraps(func)(wrapper))
    arg0 = len(args) and args[0] or None
    if recursive:
        raise NotImplementedError
        if inspect.ismodule(arg0):
            for n, f in inspect.getmembers(arg0, inspect.isfunction):
                setattr(arg0, n, decorator(f))
            for n, c in inspect.getmembers(arg0, inspect.isclass):
                dotrace(c, *args, recursive=recursive)
        elif inspect.isclass(arg0):
            for n, f in inspect.getmembers(arg0, lambda x: inspect.isfunction(x) or inspect.ismethod(x)):
                setattr(arg0, n, decorator(f))
    if callable(arg0) or type(arg0) in (classmethod, staticmethod):
        return decorator(arg0)
    elif type(arg0) == property:
        pget, pset, pdel = (None, None, None)
        if arg0.fget:
            pget = decorator(arg0.fget)
        if arg0.fset:
            pset = decorator(arg0.fset)
        if arg0.fdel:
            pdel = decorator(arg0.fdel)
        return property(pget, pset, pdel)
    else:
        return decorator