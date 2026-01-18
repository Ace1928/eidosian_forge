from ..runtime.jit import jit
from . import core as tl
from . import standard
@jit
def philox_impl(c0, c1, c2, c3, k0, k1, n_rounds: tl.constexpr=N_ROUNDS_DEFAULT):
    """
    Run `n_rounds` rounds of Philox for state (c0, c1, c2, c3) and key (k0, k1).
    """
    if c0.dtype == tl.uint32:
        PHILOX_KEY_A: tl.constexpr = 2654435769
        PHILOX_KEY_B: tl.constexpr = 3144134277
        PHILOX_ROUND_A: tl.constexpr = 3528531795
        PHILOX_ROUND_B: tl.constexpr = 3449720151
    else:
        tl.static_assert(c0.dtype == tl.uint64, 'dtype not supported in philox_impl')
        PHILOX_KEY_A: tl.constexpr = 11400714819323198485
        PHILOX_KEY_B: tl.constexpr = 13503953896175478587
        PHILOX_ROUND_A: tl.constexpr = 15197193596820024467
        PHILOX_ROUND_B: tl.constexpr = 14581110107779764567
    for _ in tl.static_range(n_rounds):
        A = PHILOX_ROUND_A
        B = PHILOX_ROUND_B
        _c0, _c2 = (c0, c2)
        c0 = tl.umulhi(B, _c2) ^ c1 ^ k0
        c2 = tl.umulhi(A, _c0) ^ c3 ^ k1
        c1 = B * _c2
        c3 = A * _c0
        k0 = k0 + PHILOX_KEY_A
        k1 = k1 + PHILOX_KEY_B
    return (c0, c1, c2, c3)