import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def kursawe(individual):
    """Kursawe multiobjective function.

    :math:`f_{\\text{Kursawe}1}(\\mathbf{x}) = \\sum_{i=1}^{N-1} -10 e^{-0.2 \\sqrt{x_i^2 + x_{i+1}^2} }`

    :math:`f_{\\text{Kursawe}2}(\\mathbf{x}) = \\sum_{i=1}^{N} |x_i|^{0.8} + 5 \\sin(x_i^3)`

    .. plot:: code/benchmarks/kursawe.py
       :width: 100 %
    """
    f1 = sum((-10 * exp(-0.2 * sqrt(x * x + y * y)) for x, y in zip(individual[:-1], individual[1:])))
    f2 = sum((abs(x) ** 0.8 + 5 * sin(x * x * x) for x in individual))
    return (f1, f2)