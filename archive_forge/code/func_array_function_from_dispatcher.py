import collections
import functools
import os
from .._utils import set_module
from .._utils._inspect import getargspec
from numpy.core._multiarray_umath import (
def array_function_from_dispatcher(implementation, module=None, verify=True, docs_from_dispatcher=True):
    """Like array_function_dispatcher, but with function arguments flipped."""

    def decorator(dispatcher):
        return array_function_dispatch(dispatcher, module, verify=verify, docs_from_dispatcher=docs_from_dispatcher)(implementation)
    return decorator