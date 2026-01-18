from ..runtime.jit import jit
from . import core as tl
from . import standard
@jit
def randn4x(seed, offset, n_rounds: tl.constexpr=N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block,
    returns 4 blocks of random :code:`float32` in :math:`\\mathcal{N}(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    u1, u2, u3, u4 = rand4x(seed, offset, n_rounds)
    n1, n2 = pair_uniform_to_normal(u1, u2)
    n3, n4 = pair_uniform_to_normal(u3, u4)
    return (n1, n2, n3, n4)