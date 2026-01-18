from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type x: Block
    :param div: the divisor
    :param div: Block
    """
    return (x + div - 1) // div