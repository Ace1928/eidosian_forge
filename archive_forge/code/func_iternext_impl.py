import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def iternext_impl(ref_type=None):
    """
    Wrap the given iternext() implementation so that it gets passed
    an _IternextResult() object easing the returning of the iternext()
    result pair.

    ref_type: a numba.targets.imputils.RefType value, the reference type used is
    that specified through the RefType enum.

    The wrapped function will be called with the following signature:
        (context, builder, sig, args, iternext_result)
    """
    if ref_type not in [x for x in RefType]:
        raise ValueError('ref_type must be an enum member of imputils.RefType')

    def outer(func):

        def wrapper(context, builder, sig, args):
            pair_type = sig.return_type
            pairobj = context.make_helper(builder, pair_type)
            func(context, builder, sig, args, _IternextResult(context, builder, pairobj))
            if ref_type == RefType.NEW:
                impl_ret = impl_ret_new_ref
            elif ref_type == RefType.BORROWED:
                impl_ret = impl_ret_borrowed
            elif ref_type == RefType.UNTRACKED:
                impl_ret = impl_ret_untracked
            else:
                raise ValueError('Unknown ref_type encountered')
            return impl_ret(context, builder, pair_type, pairobj._getvalue())
        return wrapper
    return outer