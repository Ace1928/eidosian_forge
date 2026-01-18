from ..runtime.jit import jit
from . import core as tl
from . import standard
@jit
def randn(seed, offset, n_rounds: tl.constexpr=N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block,
    returns a block of random :code:`float32` in :math:`\\mathcal{N}(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    i1, i2, _, _ = randint4x(seed, offset, n_rounds)
    u1 = uint_to_uniform_float(i1)
    u2 = uint_to_uniform_float(i2)
    n1, _ = pair_uniform_to_normal(u1, u2)
    return n1