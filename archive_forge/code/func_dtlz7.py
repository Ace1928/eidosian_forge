import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def dtlz7(ind, n_objs):
    """DTLZ7 multiobjective function. It returns a tuple of *obj* values. The
    individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
    Optimization Test Problems. CEC 2002, p. 825-830, IEEE Press, 2002.
    """
    gval = 1 + 9.0 / len(ind[n_objs - 1:]) * sum([a for a in ind[n_objs - 1:]])
    fit = [x for x in ind[:n_objs - 1]]
    fit.append((1 + gval) * (n_objs - sum([a / (1.0 + gval) * (1 + sin(3 * pi * a)) for a in ind[:n_objs - 1]])))
    return fit