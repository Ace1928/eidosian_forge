from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.complexes import (Abs, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.quantum import TensorProduct
def phase_retarder(theta=0, delta=0):
    """A phase retarder Jones matrix with retardance ``delta`` at angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.
    delta : numeric type or SymPy Symbol
        The phase difference between the fast and slow axes of the
        transmitted light.

    Returns
    =======

    SymPy Matrix :
        A Jones matrix representing the retarder.

    Examples
    ========

    A generic retarder.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import phase_retarder
    >>> theta, delta = symbols("theta, delta", real=True)
    >>> R = phase_retarder(theta, delta)
    >>> pprint(R, use_unicode=True)
    ⎡                          -ⅈ⋅δ               -ⅈ⋅δ               ⎤
    ⎢                          ─────              ─────              ⎥
    ⎢⎛ ⅈ⋅δ    2         2   ⎞    2    ⎛     ⅈ⋅δ⎞    2                ⎥
    ⎢⎝ℯ   ⋅sin (θ) + cos (θ)⎠⋅ℯ       ⎝1 - ℯ   ⎠⋅ℯ     ⋅sin(θ)⋅cos(θ)⎥
    ⎢                                                                ⎥
    ⎢            -ⅈ⋅δ                                           -ⅈ⋅δ ⎥
    ⎢            ─────                                          ─────⎥
    ⎢⎛     ⅈ⋅δ⎞    2                  ⎛ ⅈ⋅δ    2         2   ⎞    2  ⎥
    ⎣⎝1 - ℯ   ⎠⋅ℯ     ⋅sin(θ)⋅cos(θ)  ⎝ℯ   ⋅cos (θ) + sin (θ)⎠⋅ℯ     ⎦

    """
    R = Matrix([[cos(theta) ** 2 + exp(I * delta) * sin(theta) ** 2, (1 - exp(I * delta)) * cos(theta) * sin(theta)], [(1 - exp(I * delta)) * cos(theta) * sin(theta), sin(theta) ** 2 + exp(I * delta) * cos(theta) ** 2]])
    return R * exp(-I * delta / 2)