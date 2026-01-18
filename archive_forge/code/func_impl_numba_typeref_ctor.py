from collections.abc import MutableSequence
from numba.core.types import ListType
from numba.core.imputils import numba_typeref_ctor
from numba.core.dispatcher import Dispatcher
from numba.core import types, config, cgutils
from numba import njit, typeof
from numba.core.extending import (
from numba.typed import listobject
from numba.core.errors import TypingError, LoweringError
from numba.core.typing.templates import Signature
import typing as pt
@overload(numba_typeref_ctor)
def impl_numba_typeref_ctor(cls, *args):
    """Defines lowering for ``List()`` and ``List(iterable)``.

    This defines the lowering logic to instantiate either an empty typed-list
    or a typed-list initialised with values from a single iterable argument.

    Parameters
    ----------
    cls : TypeRef
        Expecting a TypeRef of a precise ListType.
    args: tuple
        A tuple that contains a single iterable (optional)

    Returns
    -------
    impl : function
        An implementation suitable for lowering the constructor call.

    See also: `redirect_type_ctor` in numba/cpython/bulitins.py
    """
    list_ty = cls.instance_type
    if not isinstance(list_ty, types.ListType):
        return
    if not list_ty.is_precise():
        msg = 'expecting a precise ListType but got {}'.format(list_ty)
        raise LoweringError(msg)
    item_type = types.TypeRef(list_ty.item_type)
    if args:
        if isinstance(args[0], types.Array) and args[0].ndim == 0:

            def impl(cls, *args):
                r = List.empty_list(item_type)
                r.append(args[0].item())
                return r
        else:

            def impl(cls, *args):
                r = List.empty_list(item_type)
                for i in args[0]:
                    r.append(i)
                return r
    else:

        def impl(cls, *args):
            return List.empty_list(item_type)
    return impl