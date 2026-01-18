from numba.core import types, errors, cgutils
from numba.core.extending import intrinsic
@intrinsic
def exception_match(typingctx, exc_value, exc_class):
    """Basically do ``isinstance(exc_value, exc_class)`` for exception objects.
    Used in ``except Exception:`` syntax.
    """
    if exc_class.exc_class is not Exception:
        msg = 'Exception matching is limited to {}'
        raise errors.UnsupportedError(msg.format(Exception))

    def codegen(context, builder, signature, args):
        return cgutils.true_bit
    restype = types.boolean
    return (restype(exc_value, exc_class), codegen)