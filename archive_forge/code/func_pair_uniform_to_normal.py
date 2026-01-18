from ..runtime.jit import jit
from . import core as tl
from . import standard
@jit
def pair_uniform_to_normal(u1, u2):
    """Box-Muller transform"""
    u1 = standard.maximum(1e-07, u1)
    th = 6.283185307179586 * u2
    r = tl.sqrt(-2.0 * tl.log(u1))
    return (r * tl.cos(th), r * tl.sin(th))