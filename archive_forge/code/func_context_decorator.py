import functools
import inspect
import warnings
import sys
from typing import Any, Callable, TypeVar, cast
def context_decorator(ctx, func):
    """
    Like contextlib.ContextDecorator.

    But with the following differences:
    1. Is done by wrapping, rather than inheritance, so it works with context
       managers that are implemented from C and thus cannot easily inherit from
       Python classes
    2. Wraps generators in the intuitive way (c.f. https://bugs.python.org/issue37743)
    3. Errors out if you try to wrap a class, because it is ambiguous whether
       or not you intended to wrap only the constructor

    The input argument can either be a context manager (in which case it must
    be a multi-shot context manager that can be directly invoked multiple times)
    or a callable that produces a context manager.
    """
    assert not (callable(ctx) and hasattr(ctx, '__enter__')), f'Passed in {ctx} is both callable and also a valid context manager (has __enter__), making it ambiguous which interface to use.  If you intended to pass a context manager factory, rewrite your call as context_decorator(lambda: ctx()); if you intended to pass a context manager directly, rewrite your call as context_decorator(lambda: ctx)'
    if not callable(ctx):

        def ctx_factory():
            return ctx
    else:
        ctx_factory = ctx
    if inspect.isclass(func):
        raise RuntimeError('Cannot decorate classes; it is ambiguous whether or not only the constructor or all methods should have the context manager applied; additionally, decorating a class at definition-site will prevent use of the identifier as a conventional type.  To specify which methods to decorate, decorate each of them individually.')
    if inspect.isgeneratorfunction(func):
        return _wrap_generator(ctx_factory, func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        with ctx_factory():
            return func(*args, **kwargs)
    return decorate_context