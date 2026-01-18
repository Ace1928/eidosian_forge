import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def dtlz1(individual, obj):
    """DTLZ1 multiobjective function. It returns a tuple of *obj* values.
    The individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
    Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.

    :math:`g(\\mathbf{x}_m) = 100\\left(|\\mathbf{x}_m| + \\sum_{x_i \\in \\mathbf{x}_m}\\left((x_i - 0.5)^2 - \\cos(20\\pi(x_i - 0.5))\\right)\\right)`

    :math:`f_{\\text{DTLZ1}1}(\\mathbf{x}) = \\frac{1}{2} (1 + g(\\mathbf{x}_m)) \\prod_{i=1}^{m-1}x_i`

    :math:`f_{\\text{DTLZ1}2}(\\mathbf{x}) = \\frac{1}{2} (1 + g(\\mathbf{x}_m)) (1-x_{m-1}) \\prod_{i=1}^{m-2}x_i`

    :math:`\\ldots`

    :math:`f_{\\text{DTLZ1}m-1}(\\mathbf{x}) = \\frac{1}{2} (1 + g(\\mathbf{x}_m)) (1 - x_2) x_1`

    :math:`f_{\\text{DTLZ1}m}(\\mathbf{x}) = \\frac{1}{2} (1 - x_1)(1 + g(\\mathbf{x}_m))`

    Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m` is a
    vector of the remaining attributes :math:`[x_m~\\ldots~x_n]` of the
    individual in :math:`n > m` dimensions.

    """
    g = 100 * (len(individual[obj - 1:]) + sum(((xi - 0.5) ** 2 - cos(20 * pi * (xi - 0.5)) for xi in individual[obj - 1:])))
    f = [0.5 * reduce(mul, individual[:obj - 1], 1) * (1 + g)]
    f.extend((0.5 * reduce(mul, individual[:m], 1) * (1 - individual[m]) * (1 + g) for m in reversed(range(obj - 1))))
    return f