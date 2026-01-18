from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableMatrix
from sympy.utilities.misc import as_int
def gaunt(l_1, l_2, l_3, m_1, m_2, m_3, prec=None):
    """
    Calculate the Gaunt coefficient.

    Explanation
    ===========

    The Gaunt coefficient is defined as the integral over three
    spherical harmonics:

    .. math::

        \\begin{aligned}
        \\operatorname{Gaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\\int Y_{l_1,m_1}(\\Omega)
         Y_{l_2,m_2}(\\Omega) Y_{l_3,m_3}(\\Omega) \\,d\\Omega \\\\
        &=\\sqrt{\\frac{(2l_1+1)(2l_2+1)(2l_3+1)}{4\\pi}}
         \\operatorname{Wigner3j}(l_1,l_2,l_3,0,0,0)
         \\operatorname{Wigner3j}(l_1,l_2,l_3,m_1,m_2,m_3)
        \\end{aligned}

    Parameters
    ==========

    l_1, l_2, l_3, m_1, m_2, m_3 :
        Integer.
    prec - precision, default: ``None``.
        Providing a precision can
        drastically speed up the calculation.

    Returns
    =======

    Rational number times the square root of a rational number
    (if ``prec=None``), or real number if a precision is given.

    Examples
    ========

    >>> from sympy.physics.wigner import gaunt
    >>> gaunt(1,0,1,1,0,-1)
    -1/(2*sqrt(pi))
    >>> gaunt(1000,1000,1200,9,3,-12).n(64)
    0.00689500421922113448...

    It is an error to use non-integer values for `l` and `m`::

        sage: gaunt(1.2,0,1.2,0,0,0)
        Traceback (most recent call last):
        ...
        ValueError: l values must be integer
        sage: gaunt(1,0,1,1.1,0,-1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer

    Notes
    =====

    The Gaunt coefficient obeys the following symmetry rules:

    - invariant under any permutation of the columns

      .. math::
        \\begin{aligned}
          Y(l_1,l_2,l_3,m_1,m_2,m_3)
          &=Y(l_3,l_1,l_2,m_3,m_1,m_2) \\\\
          &=Y(l_2,l_3,l_1,m_2,m_3,m_1) \\\\
          &=Y(l_3,l_2,l_1,m_3,m_2,m_1) \\\\
          &=Y(l_1,l_3,l_2,m_1,m_3,m_2) \\\\
          &=Y(l_2,l_1,l_3,m_2,m_1,m_3)
        \\end{aligned}

    - invariant under space inflection, i.e.

      .. math::
          Y(l_1,l_2,l_3,m_1,m_2,m_3)
          =Y(l_1,l_2,l_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 Regge symmetries as inherited
      for the `3j` symbols [Regge58]_

    - zero for `l_1`, `l_2`, `l_3` not fulfilling triangle relation

    - zero for violating any one of the conditions: `l_1 \\ge |m_1|`,
      `l_2 \\ge |m_2|`, `l_3 \\ge |m_3|`

    - non-zero only for an even sum of the `l_i`, i.e.
      `L = l_1 + l_2 + l_3 = 2n` for `n` in `\\mathbb{N}`

    Algorithms
    ==========

    This function uses the algorithm of [Liberatodebrito82]_ to
    calculate the value of the Gaunt coefficient exactly. Note that
    the formula contains alternating sums over large factorials and is
    therefore unsuitable for finite precision arithmetic and only
    useful for a computer algebra system [Rasch03]_.

    Authors
    =======

    Jens Rasch (2009-03-24): initial version for Sage.
    """
    l_1, l_2, l_3, m_1, m_2, m_3 = [as_int(i) for i in (l_1, l_2, l_3, m_1, m_2, m_3)]
    if l_1 + l_2 - l_3 < 0:
        return S.Zero
    if l_1 - l_2 + l_3 < 0:
        return S.Zero
    if -l_1 + l_2 + l_3 < 0:
        return S.Zero
    if m_1 + m_2 + m_3 != 0:
        return S.Zero
    if abs(m_1) > l_1 or abs(m_2) > l_2 or abs(m_3) > l_3:
        return S.Zero
    bigL, remL = divmod(l_1 + l_2 + l_3, 2)
    if remL % 2:
        return S.Zero
    imin = max(-l_3 + l_1 + m_2, -l_3 + l_2 - m_1, 0)
    imax = min(l_2 + m_2, l_1 - m_1, l_1 + l_2 - l_3)
    _calc_factlist(max(l_1 + l_2 + l_3 + 1, imax + 1))
    ressqrt = sqrt((2 * l_1 + 1) * (2 * l_2 + 1) * (2 * l_3 + 1) * _Factlist[l_1 - m_1] * _Factlist[l_1 + m_1] * _Factlist[l_2 - m_2] * _Factlist[l_2 + m_2] * _Factlist[l_3 - m_3] * _Factlist[l_3 + m_3] / (4 * pi))
    prefac = Integer(_Factlist[bigL] * _Factlist[l_2 - l_1 + l_3] * _Factlist[l_1 - l_2 + l_3] * _Factlist[l_1 + l_2 - l_3]) / _Factlist[2 * bigL + 1] / (_Factlist[bigL - l_1] * _Factlist[bigL - l_2] * _Factlist[bigL - l_3])
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * _Factlist[ii + l_3 - l_1 - m_2] * _Factlist[l_2 + m_2 - ii] * _Factlist[l_1 - ii - m_1] * _Factlist[ii + l_3 - l_2 + m_1] * _Factlist[l_1 + l_2 - l_3 - ii]
        sumres = sumres + Integer((-1) ** ii) / den
    res = ressqrt * prefac * sumres * Integer((-1) ** (bigL + l_3 + m_1 - m_2))
    if prec is not None:
        res = res.n(prec)
    return res