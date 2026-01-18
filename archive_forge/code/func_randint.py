from ..runtime.jit import jit
from . import core as tl
from . import standard
@jit
def randint(seed, offset, n_rounds: tl.constexpr=N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block, returns a single
    block of random :code:`int32`.

    If you need multiple streams of random numbers,
    using `randint4x` is likely to be faster than calling `randint` 4 times.

    :param seed: The seed for generating random numbers.
    :param offset: The offsets to generate random numbers for.
    """
    ret, _, _, _ = randint4x(seed, offset, n_rounds)
    return ret