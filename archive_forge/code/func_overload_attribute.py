import os
import uuid
import weakref
import collections
import functools
import numba
from numba.core import types, errors, utils, config
from numba.core.typing.typeof import typeof_impl  # noqa: F401
from numba.core.typing.asnumbatype import as_numba_type  # noqa: F401
from numba.core.typing.templates import infer, infer_getattr  # noqa: F401
from numba.core.imputils import (  # noqa: F401
from numba.core.datamodel import models   # noqa: F401
from numba.core.datamodel import register_default as register_model  # noqa: F401, E501
from numba.core.pythonapi import box, unbox, reflect, NativeValue  # noqa: F401
from numba._helperlib import _import_cython_function  # noqa: F401
from numba.core.serialize import ReduceMixin
def overload_attribute(typ, attr, **kwargs):
    """
    A decorator marking the decorated function as typing and implementing
    attribute *attr* for the given Numba type in nopython mode.

    *kwargs* are passed to the underlying `@overload` call.

    Here is an example implementing .nbytes for array types::

        @overload_attribute(types.Array, 'nbytes')
        def array_nbytes(arr):
            def get(arr):
                return arr.size * arr.itemsize
            return get
    """
    from numba.core.typing.templates import make_overload_attribute_template

    def decorate(overload_func):
        template = make_overload_attribute_template(typ, attr, overload_func, inline=kwargs.get('inline', 'never'))
        infer_getattr(template)
        overload(overload_func, **kwargs)(overload_func)
        return overload_func
    return decorate