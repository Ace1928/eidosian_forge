import collections
from importlib import util
import inspect
import sys
def inspect_getargspec(func):
    """getargspec based on fully vendored getfullargspec from Python 3.3."""
    if inspect.ismethod(func):
        func = func.__func__
    if not inspect.isfunction(func):
        raise TypeError(f'{func!r} is not a Python function')
    co = func.__code__
    if not inspect.iscode(co):
        raise TypeError(f'{co!r} is not a code object')
    nargs = co.co_argcount
    names = co.co_varnames
    nkwargs = co.co_kwonlyargcount
    args = list(names[:nargs])
    nargs += nkwargs
    varargs = None
    if co.co_flags & inspect.CO_VARARGS:
        varargs = co.co_varnames[nargs]
        nargs = nargs + 1
    varkw = None
    if co.co_flags & inspect.CO_VARKEYWORDS:
        varkw = co.co_varnames[nargs]
    return ArgSpec(args, varargs, varkw, func.__defaults__)