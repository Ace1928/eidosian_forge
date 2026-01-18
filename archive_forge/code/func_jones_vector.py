from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.complexes import (Abs, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.quantum import TensorProduct
def jones_vector(psi, chi):
    """A Jones vector corresponding to a polarization ellipse with `psi` tilt,
    and `chi` circularity.

    Parameters
    ==========

    psi : numeric type or SymPy Symbol
        The tilt of the polarization relative to the `x` axis.

    chi : numeric type or SymPy Symbol
        The angle adjacent to the mayor axis of the polarization ellipse.


    Returns
    =======

    Matrix :
        A Jones vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import jones_vector
    >>> psi, chi = symbols("psi, chi", real=True)

    A general Jones vector.

    >>> pprint(jones_vector(psi, chi), use_unicode=True)
    ⎡-ⅈ⋅sin(χ)⋅sin(ψ) + cos(χ)⋅cos(ψ)⎤
    ⎢                                ⎥
    ⎣ⅈ⋅sin(χ)⋅cos(ψ) + sin(ψ)⋅cos(χ) ⎦

    Horizontal polarization.

    >>> pprint(jones_vector(0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎣0⎦

    Vertical polarization.

    >>> pprint(jones_vector(pi/2, 0), use_unicode=True)
    ⎡0⎤
    ⎢ ⎥
    ⎣1⎦

    Diagonal polarization.

    >>> pprint(jones_vector(pi/4, 0), use_unicode=True)
    ⎡√2⎤
    ⎢──⎥
    ⎢2 ⎥
    ⎢  ⎥
    ⎢√2⎥
    ⎢──⎥
    ⎣2 ⎦

    Anti-diagonal polarization.

    >>> pprint(jones_vector(-pi/4, 0), use_unicode=True)
    ⎡ √2 ⎤
    ⎢ ── ⎥
    ⎢ 2  ⎥
    ⎢    ⎥
    ⎢-√2 ⎥
    ⎢────⎥
    ⎣ 2  ⎦

    Right-hand circular polarization.

    >>> pprint(jones_vector(0, pi/4), use_unicode=True)
    ⎡ √2 ⎤
    ⎢ ── ⎥
    ⎢ 2  ⎥
    ⎢    ⎥
    ⎢√2⋅ⅈ⎥
    ⎢────⎥
    ⎣ 2  ⎦

    Left-hand circular polarization.

    >>> pprint(jones_vector(0, -pi/4), use_unicode=True)
    ⎡  √2  ⎤
    ⎢  ──  ⎥
    ⎢  2   ⎥
    ⎢      ⎥
    ⎢-√2⋅ⅈ ⎥
    ⎢──────⎥
    ⎣  2   ⎦

    """
    return Matrix([-I * sin(chi) * sin(psi) + cos(chi) * cos(psi), I * sin(chi) * cos(psi) + sin(psi) * cos(chi)])