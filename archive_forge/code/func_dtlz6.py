import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def dtlz6(ind, n_objs):
    """DTLZ6 multiobjective function. It returns a tuple of *obj* values. The
    individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
    Optimization Test Problems. CEC 2002, p. 825-830, IEEE Press, 2002.
    """
    gval = sum([a ** 0.1 for a in ind[n_objs - 1:]])
    theta = lambda x: pi / (4.0 * (1 + gval)) * (1 + 2 * gval * x)
    fit = [(1 + gval) * cos(pi / 2.0 * ind[0]) * reduce(lambda x, y: x * y, [cos(theta(a)) for a in ind[1:]])]
    for m in reversed(range(1, n_objs)):
        if m == 1:
            fit.append((1 + gval) * sin(pi / 2.0 * ind[0]))
        else:
            fit.append((1 + gval) * cos(pi / 2.0 * ind[0]) * reduce(lambda x, y: x * y, [cos(theta(a)) for a in ind[1:m - 1]], 1) * sin(theta(ind[m - 1])))
    return fit