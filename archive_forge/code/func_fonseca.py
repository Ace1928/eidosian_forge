import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def fonseca(individual):
    """Fonseca and Fleming's multiobjective function.
    From: C. M. Fonseca and P. J. Fleming, "Multiobjective optimization and
    multiple constraint handling with evolutionary algorithms -- Part II:
    Application example", IEEE Transactions on Systems, Man and Cybernetics,
    1998.

    :math:`f_{\\text{Fonseca}1}(\\mathbf{x}) = 1 - e^{-\\sum_{i=1}^{3}(x_i - \\frac{1}{\\sqrt{3}})^2}`

    :math:`f_{\\text{Fonseca}2}(\\mathbf{x}) = 1 - e^{-\\sum_{i=1}^{3}(x_i + \\frac{1}{\\sqrt{3}})^2}`
    """
    f_1 = 1 - exp(-sum(((xi - 1 / sqrt(3)) ** 2 for xi in individual[:3])))
    f_2 = 1 - exp(-sum(((xi + 1 / sqrt(3)) ** 2 for xi in individual[:3])))
    return (f_1, f_2)