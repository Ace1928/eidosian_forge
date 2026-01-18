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
def include_path():
    """Returns the C include directory path.
    """
    include_dir = os.path.dirname(os.path.dirname(numba.__file__))
    path = os.path.abspath(include_dir)
    return path