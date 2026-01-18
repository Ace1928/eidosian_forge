from numba.core.extending import overload
from numba.core import types
from numba.misc.special import literally, literal_unroll
from numba.core.errors import TypingError
@overload(literal_unroll)
def literal_unroll_impl(container):
    if isinstance(container, types.Poison):
        m = f'Invalid use of non-Literal type in literal_unroll({container})'
        raise TypingError(m)

    def impl(container):
        return container
    return impl