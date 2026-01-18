from sympy.core.numbers import I
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.utilities.decorator import deprecated
@deprecated('\n    The sympy.physics.matrices.mdft method is deprecated. Use\n    sympy.DFT(n).as_explicit() instead.\n    ', deprecated_since_version='1.9', active_deprecations_target='deprecated-physics-mdft')
def mdft(n):
    """
    .. deprecated:: 1.9

       Use DFT from sympy.matrices.expressions.fourier instead.

       To get identical behavior to ``mdft(n)``, use ``DFT(n).as_explicit()``.
    """
    from sympy.matrices.expressions.fourier import DFT
    return DFT(n).as_mutable()