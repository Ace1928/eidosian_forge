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
def intrinsic(*args, **kwargs):
    """
    A decorator marking the decorated function as typing and implementing
    *func* in nopython mode using the llvmlite IRBuilder API.  This is an escape
    hatch for expert users to build custom LLVM IR that will be inlined to
    the caller.

    The first argument to *func* is the typing context.  The rest of the
    arguments corresponds to the type of arguments of the decorated function.
    These arguments are also used as the formal argument of the decorated
    function.  If *func* has the signature ``foo(typing_context, arg0, arg1)``,
    the decorated function will have the signature ``foo(arg0, arg1)``.

    The return values of *func* should be a 2-tuple of expected type signature,
    and a code-generation function that will passed to ``lower_builtin``.
    For unsupported operation, return None.

    Here is an example implementing a ``cast_int_to_byte_ptr`` that cast
    any integer to a byte pointer::

        @intrinsic
        def cast_int_to_byte_ptr(typingctx, src):
            # check for accepted types
            if isinstance(src, types.Integer):
                # create the expected type signature
                result_type = types.CPointer(types.uint8)
                sig = result_type(types.uintp)
                # defines the custom code generation
                def codegen(context, builder, signature, args):
                    # llvm IRBuilder code here
                    [src] = args
                    rtype = signature.return_type
                    llrtype = context.get_value_type(rtype)
                    return builder.inttoptr(src, llrtype)
                return sig, codegen
    """

    def _intrinsic(func):
        name = getattr(func, '__name__', str(func))
        llc = _Intrinsic(name, func, **kwargs)
        llc._register()
        return llc
    if not kwargs:
        return _intrinsic(*args)
    else:

        def wrapper(func):
            return _intrinsic(func)
        return wrapper