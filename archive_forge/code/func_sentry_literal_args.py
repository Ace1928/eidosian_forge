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
def sentry_literal_args(pysig, literal_args, args, kwargs):
    """Ensures that the given argument types (in *args* and *kwargs*) are
    literally typed for a function with the python signature *pysig* and the
    list of literal argument names in *literal_args*.

    Alternatively, this is the same as::

        SentryLiteralArgs(literal_args).for_pysig(pysig).bind(*args, **kwargs)
    """
    boundargs = pysig.bind(*args, **kwargs)
    request_pos = set()
    missing = False
    for i, (k, v) in enumerate(boundargs.arguments.items()):
        if k in literal_args:
            request_pos.add(i)
            if not isinstance(v, types.Literal):
                missing = True
    if missing:
        e = errors.ForceLiteralArg(request_pos)

        def folded(args, kwargs):
            out = pysig.bind(*args, **kwargs).arguments.values()
            return tuple(out)
        raise e.bind_fold_arguments(folded)