import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def rastrigin_skew(individual):
    """Skewed Rastrigin test objective function.

     :math:`f_{\\text{RastSkew}}(\\mathbf{x}) = 10N + \\sum_{i=1}^N \\left(y_i^2 - 10 \\cos(2\\pi x_i)\\right)`

     :math:`\\text{with } y_i = \\
                            \\begin{cases} \\
                                10\\cdot x_i & \\text{ if } x_i > 0,\\\\ \\
                                x_i & \\text{ otherwise } \\
                            \\end{cases}`
    """
    N = len(individual)
    return (10 * N + sum(((10 * x if x > 0 else x) ** 2 - 10 * cos(2 * pi * (10 * x if x > 0 else x)) for x in individual)),)