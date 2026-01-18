import inspect
import types
import typing as t
from functools import update_wrapper
from gettext import gettext as _
from .core import Argument
from .core import Command
from .core import Context
from .core import Group
from .core import Option
from .core import Parameter
from .globals import get_current_context
from .utils import echo
def make_pass_decorator(object_type: t.Type[T], ensure: bool=False) -> t.Callable[['t.Callable[te.Concatenate[T, P], R]'], 't.Callable[P, R]']:
    """Given an object type this creates a decorator that will work
    similar to :func:`pass_obj` but instead of passing the object of the
    current context, it will find the innermost context of type
    :func:`object_type`.

    This generates a decorator that works roughly like this::

        from functools import update_wrapper

        def decorator(f):
            @pass_context
            def new_func(ctx, *args, **kwargs):
                obj = ctx.find_object(object_type)
                return ctx.invoke(f, obj, *args, **kwargs)
            return update_wrapper(new_func, f)
        return decorator

    :param object_type: the type of the object to pass.
    :param ensure: if set to `True`, a new object will be created and
                   remembered on the context if it's not there yet.
    """

    def decorator(f: 't.Callable[te.Concatenate[T, P], R]') -> 't.Callable[P, R]':

        def new_func(*args: 'P.args', **kwargs: 'P.kwargs') -> 'R':
            ctx = get_current_context()
            obj: t.Optional[T]
            if ensure:
                obj = ctx.ensure_object(object_type)
            else:
                obj = ctx.find_object(object_type)
            if obj is None:
                raise RuntimeError(f'Managed to invoke callback without a context object of type {object_type.__name__!r} existing.')
            return ctx.invoke(f, obj, *args, **kwargs)
        return update_wrapper(new_func, f)
    return decorator