from ..runtime.jit import jit
from . import core as tl
from . import standard
@jit
def philox(seed, c0, c1, c2, c3, n_rounds: tl.constexpr=N_ROUNDS_DEFAULT):
    seed = seed.to(tl.uint64)
    if tl.constexpr(c0.dtype.primitive_bitwidth) == 32:
        int_dtype = tl.uint32
        seed_hi = (seed >> 32 & 4294967295).to(tl.uint32)
        seed_lo = (seed & 4294967295).to(tl.uint32)
    else:
        tl.static_assert(tl.constexpr(c0.dtype.primitive_bitwidth) == 64, 'bitwidth not supported in philox')
        int_dtype = tl.uint64
        seed_hi = 0
        seed_lo = seed
    c0 = c0.to(int_dtype, bitcast=True)
    c1 = c1.to(int_dtype, bitcast=True)
    c2 = c2.to(int_dtype, bitcast=True)
    c3 = c3.to(int_dtype, bitcast=True)
    return philox_impl(c0, c1, c2, c3, seed_lo, seed_hi, n_rounds)