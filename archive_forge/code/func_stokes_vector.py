from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.complexes import (Abs, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.quantum import TensorProduct
def stokes_vector(psi, chi, p=1, I=1):
    """A Stokes vector corresponding to a polarization ellipse with ``psi``
    tilt, and ``chi`` circularity.

    Parameters
    ==========

    psi : numeric type or SymPy Symbol
        The tilt of the polarization relative to the ``x`` axis.
    chi : numeric type or SymPy Symbol
        The angle adjacent to the mayor axis of the polarization ellipse.
    p : numeric type or SymPy Symbol
        The degree of polarization.
    I : numeric type or SymPy Symbol
        The intensity of the field.


    Returns
    =======

    Matrix :
        A Stokes vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import stokes_vector
    >>> psi, chi, p, I = symbols("psi, chi, p, I", real=True)
    >>> pprint(stokes_vector(psi, chi, p, I), use_unicode=True)
    ⎡          I          ⎤
    ⎢                     ⎥
    ⎢I⋅p⋅cos(2⋅χ)⋅cos(2⋅ψ)⎥
    ⎢                     ⎥
    ⎢I⋅p⋅sin(2⋅ψ)⋅cos(2⋅χ)⎥
    ⎢                     ⎥
    ⎣    I⋅p⋅sin(2⋅χ)     ⎦


    Horizontal polarization

    >>> pprint(stokes_vector(0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢1⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣0⎦

    Vertical polarization

    >>> pprint(stokes_vector(pi/2, 0), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢-1⎥
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎣0 ⎦

    Diagonal polarization

    >>> pprint(stokes_vector(pi/4, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢1⎥
    ⎢ ⎥
    ⎣0⎦

    Anti-diagonal polarization

    >>> pprint(stokes_vector(-pi/4, 0), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎢-1⎥
    ⎢  ⎥
    ⎣0 ⎦

    Right-hand circular polarization

    >>> pprint(stokes_vector(0, pi/4), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣1⎦

    Left-hand circular polarization

    >>> pprint(stokes_vector(0, -pi/4), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎣-1⎦

    Unpolarized light

    >>> pprint(stokes_vector(0, 0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣0⎦

    """
    S0 = I
    S1 = I * p * cos(2 * psi) * cos(2 * chi)
    S2 = I * p * sin(2 * psi) * cos(2 * chi)
    S3 = I * p * sin(2 * chi)
    return Matrix([S0, S1, S2, S3])