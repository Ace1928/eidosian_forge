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
def real_gaunt(l_1, l_2, l_3, m_1, m_2, m_3, prec=None):
    """
    Calculate the real Gaunt coefficient.

    Explanation
    ===========

    The real Gaunt coefficient is defined as the integral over three
    real spherical harmonics:

    .. math::
        \\begin{aligned}
        \\operatorname{RealGaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\\int Z^{m_1}_{l_1}(\\Omega)
         Z^{m_2}_{l_2}(\\Omega) Z^{m_3}_{l_3}(\\Omega) \\,d\\Omega \\\\
        \\end{aligned}

    Alternatively, it can be defined in terms of the standard Gaunt
    coefficient by relating the real spherical harmonics to the standard
    spherical harmonics via a unitary transformation `U`, i.e.
    `Z^{m}_{l}(\\Omega)=\\sum_{m'}U^{m}_{m'}Y^{m'}_{l}(\\Omega)` [Homeier96]_.
    The real Gaunt coefficient is then defined as

    .. math::
        \\begin{aligned}
        \\operatorname{RealGaunt}(l_1,l_2,l_3,m_1,m_2,m_3)
        &=\\int Z^{m_1}_{l_1}(\\Omega)
         Z^{m_2}_{l_2}(\\Omega) Z^{m_3}_{l_3}(\\Omega) \\,d\\Omega \\\\
        &=\\sum_{m'_1 m'_2 m'_3} U^{m_1}_{m'_1}U^{m_2}_{m'_2}U^{m_3}_{m'_3}
         \\operatorname{Gaunt}(l_1,l_2,l_3,m'_1,m'_2,m'_3)
        \\end{aligned}

    The unitary matrix `U` has components

    .. math::
        \\begin{aligned}
        U^m_{m'} = \\delta_{|m||m'|}*(\\delta_{m'0}\\delta_{m0} + \\frac{1}{\\sqrt{2}}\\big[\\Theta(m)
        \\big(\\delta_{m'm}+(-1)^{m'}\\delta_{m'-m}\\big)+i\\Theta(-m)\\big((-1)^{-m}
        \\delta_{m'-m}-\\delta_{m'm}*(-1)^{m'-m}\\big)\\big])
        \\end{aligned}

    where `\\delta_{ij}` is the Kronecker delta symbol and `\\Theta` is a step
    function defined as

    .. math::
        \\begin{aligned}
        \\Theta(x) = \\begin{cases} 1 \\,\\text{for}\\, x > 0 \\\\ 0 \\,\\text{for}\\, x \\leq 0 \\end{cases}
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

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy.physics.wigner import real_gaunt
    >>> real_gaunt(2,2,4,-1,-1,0)
    -2/(7*sqrt(pi))
    >>> real_gaunt(10,10,20,-9,-9,0).n(64)
    -0.00002480019791932209313156167...

    It is an error to use non-integer values for `l` and `m`::
        real_gaunt(2.8,0.5,1.3,0,0,0)
        Traceback (most recent call last):
        ...
        ValueError: l values must be integer
        real_gaunt(2,2,4,0.7,1,-3.4)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer

    Notes
    =====

    The real Gaunt coefficient inherits from the standard Gaunt coefficient,
    the invariance under any permutation of the pairs `(l_i, m_i)` and the
    requirement that the sum of the `l_i` be even to yield a non-zero value.
    It also obeys the following symmetry rules:

    - zero for `l_1`, `l_2`, `l_3` not fulfiling the condition
      `l_1 \\in \\{l_{\\text{max}}, l_{\\text{max}}-2, \\ldots, l_{\\text{min}}\\}`,
      where `l_{\\text{max}} = l_2+l_3`,

      .. math::
          \\begin{aligned}
          l_{\\text{min}} = \\begin{cases} \\kappa(l_2, l_3, m_2, m_3) & \\text{if}\\,
          \\kappa(l_2, l_3, m_2, m_3) + l_{\\text{max}}\\, \\text{is even} \\\\
          \\kappa(l_2, l_3, m_2, m_3)+1 & \\text{if}\\, \\kappa(l_2, l_3, m_2, m_3) +
          l_{\\text{max}}\\, \\text{is odd}\\end{cases}
          \\end{aligned}

      and `\\kappa(l_2, l_3, m_2, m_3) = \\max{\\big(|l_2-l_3|, \\min{\\big(|m_2+m_3|,
      |m_2-m_3|\\big)}\\big)}`

    - zero for an odd number of negative `m_i`

    Algorithms
    ==========

    This function uses the algorithms of [Homeier96]_ and [Rasch03]_ to
    calculate the value of the real Gaunt coefficient exactly. Note that
    the formula used in [Rasch03]_ contains alternating sums over large
    factorials and is therefore unsuitable for finite precision arithmetic
    and only useful for a computer algebra system [Rasch03]_. However, this
    function can in principle use any algorithm that computes the Gaunt
    coefficient, so it is suitable for finite precision arithmetic in so far
    as the algorithm which computes the Gaunt coefficient is.
    """
    l_1, l_2, l_3, m_1, m_2, m_3 = [as_int(i) for i in (l_1, l_2, l_3, m_1, m_2, m_3)]
    if sum((1 for i in (m_1, m_2, m_3) if i < 0)) % 2:
        return S.Zero
    if (l_1 + l_2 + l_3) % 2:
        return S.Zero
    lmax = l_2 + l_3
    lmin = max(abs(l_2 - l_3), min(abs(m_2 + m_3), abs(m_2 - m_3)))
    if (lmin + lmax) % 2:
        lmin += 1
    if lmin not in range(lmax, lmin - 2, -2):
        return S.Zero
    kron_del = lambda i, j: 1 if i == j else 0
    s = lambda e: -1 if e % 2 else 1
    A = lambda a, b: -kron_del(a, b) * s(a - b) + kron_del(a, -b) * s(b) if b < 0 else 0
    B = lambda a, b: kron_del(a, b) + kron_del(a, -b) * s(a) if b > 0 else 0
    C = lambda a, b: kron_del(abs(a), abs(b)) * (kron_del(a, 0) * kron_del(b, 0) + (B(a, b) + I * A(a, b)) / sqrt(2))
    ugnt = 0
    for i in range(-l_1, l_1 + 1):
        U1 = C(i, m_1)
        for j in range(-l_2, l_2 + 1):
            U2 = C(j, m_2)
            U3 = C(-i - j, m_3)
            ugnt = ugnt + re(U1 * U2 * U3) * gaunt(l_1, l_2, l_3, i, j, -i - j)
    if prec is not None:
        ugnt = ugnt.n(prec)
    return ugnt