import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def rastrigin_scaled(individual):
    """Scaled Rastrigin test objective function.

    :math:`f_{\\text{RastScaled}}(\\mathbf{x}) = 10N + \\sum_{i=1}^N \\
        \\left(10^{\\left(\\frac{i-1}{N-1}\\right)} x_i \\right)^2 - \\
        10\\cos\\left(2\\pi 10^{\\left(\\frac{i-1}{N-1}\\right)} x_i \\right)`
    """
    N = len(individual)
    return (10 * N + sum(((10 ** (i / (N - 1)) * x) ** 2 - 10 * cos(2 * pi * 10 ** (i / (N - 1)) * x) for i, x in enumerate(individual))),)