from typing import Type
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import ImmutableMatrix, eye
from sympy.matrices.expressions import MatMul, MatAdd
from sympy.polys import Poly, rootof
from sympy.polys.polyroots import roots
from sympy.polys.polytools import (cancel, degree)
from sympy.series import limit
from mpmath.libmp.libmpf import prec_to_dps
def backward_diff(tf, sample_per):
    """
        Returns falling coefficients of H(z) from numerator and denominator.
        Where H(z) is the corresponding discretized transfer function,
        discretized with the backward difference transform method.
        H(z) is obtained from the continuous transfer function H(s)
        by substituting s(z) =  (z-1)/(T*z) into H(s), where T is the
        sample period.
        Coefficients are falling, i.e. H(z) = (az+b)/(cz+d) is returned
        as [a, b], [c, d].

        Examples
        ========

        >>> from sympy.physics.control.lti import TransferFunction, backward_diff
        >>> from sympy.abc import s, L, R, T
        >>> tf = TransferFunction(1, s*L + R, s)
        >>> numZ, denZ = backward_diff(tf, T)
        >>> numZ
        [T, 0]
        >>> denZ
        [L + R*T, -L]
        """
    T = sample_per
    s = tf.var
    z = s
    np = tf.num.as_poly(s).all_coeffs()
    dp = tf.den.as_poly(s).all_coeffs()
    N = max(len(np), len(dp)) - 1
    num = Add(*[T ** (N - i) * c * (z - 1) ** i * z ** (N - i) for c, i in zip(np[::-1], range(len(np)))])
    den = Add(*[T ** (N - i) * c * (z - 1) ** i * z ** (N - i) for c, i in zip(dp[::-1], range(len(dp)))])
    num_coefs = num.as_poly(z).all_coeffs()
    den_coefs = den.as_poly(z).all_coeffs()
    return (num_coefs, den_coefs)