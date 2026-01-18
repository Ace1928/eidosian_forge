from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.complexes import (Abs, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.quantum import TensorProduct
def quarter_wave_retarder(theta):
    """A quarter-wave retarder Jones matrix at angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the retarder.

    Examples
    ========

    A generic quarter-wave plate.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import quarter_wave_retarder
    >>> theta= symbols("theta", real=True)
    >>> QWP = quarter_wave_retarder(theta)
    >>> pprint(QWP, use_unicode=True)
    ⎡                       -ⅈ⋅π            -ⅈ⋅π               ⎤
    ⎢                       ─────           ─────              ⎥
    ⎢⎛     2         2   ⎞    4               4                ⎥
    ⎢⎝ⅈ⋅sin (θ) + cos (θ)⎠⋅ℯ       (1 - ⅈ)⋅ℯ     ⋅sin(θ)⋅cos(θ)⎥
    ⎢                                                          ⎥
    ⎢         -ⅈ⋅π                                        -ⅈ⋅π ⎥
    ⎢         ─────                                       ─────⎥
    ⎢           4                  ⎛   2           2   ⎞    4  ⎥
    ⎣(1 - ⅈ)⋅ℯ     ⋅sin(θ)⋅cos(θ)  ⎝sin (θ) + ⅈ⋅cos (θ)⎠⋅ℯ     ⎦

    """
    return phase_retarder(theta, pi / 2)