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
def overload_classmethod(typ, attr, **kwargs):
    """
    A decorator marking the decorated function as typing and implementing
    classmethod *attr* for the given Numba type in nopython mode.


    Similar to ``overload_method``.


    Here is an example implementing a classmethod on the Array type to call
    ``np.arange()``::

        @overload_classmethod(types.Array, "make")
        def ov_make(cls, nitems):
            def impl(cls, nitems):
                return np.arange(nitems)
            return impl

    The above code will allow the following to work in jit-compiled code::

        @njit
        def foo(n):
            return types.Array.make(n)
    """
    return _overload_method_common(types.TypeRef(typ), attr, **kwargs)