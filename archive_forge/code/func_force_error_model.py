import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
@contextlib.contextmanager
def force_error_model(context, model_name='numpy'):
    """
    Temporarily change the context's error model.
    """
    from numba.core import callconv
    old_error_model = context.error_model
    context.error_model = callconv.create_error_model(model_name, context)
    try:
        yield
    finally:
        context.error_model = old_error_model