from numba.core import types, typing
from numba.core.cgutils import unpack_tuple
from numba.core.extending import intrinsic
from numba.core.imputils import impl_ret_new_ref
from numba.core.errors import RequireLiteralValue, TypingError
from numba.cpython.unsafe.tuple import tuple_setitem
@intrinsic
def empty_inferred(typingctx, shape):
    """A version of numpy.empty whose dtype is inferred by the type system.

    Expects `shape` to be a int-tuple.

    There is special logic in the type-inferencer to handle the "refine"-ing
    of undefined dtype.
    """
    from numba.np.arrayobj import _empty_nd_impl

    def codegen(context, builder, signature, args):
        arrty = signature.return_type
        assert arrty.is_precise()
        shapes = unpack_tuple(builder, args[0])
        res = _empty_nd_impl(context, builder, arrty, shapes)
        return impl_ret_new_ref(context, builder, arrty, res._getvalue())
    nd = len(shape)
    array_ty = types.Array(ndim=nd, layout='C', dtype=types.undefined)
    sig = array_ty(shape)
    return (sig, codegen)