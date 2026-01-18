import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def rastrigin(individual):
    """Rastrigin test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \\in [-5.12, 5.12]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = 10N + \\sum_{i=1}^N x_i^2 - 10 \\cos(2\\pi x_i)`

    .. plot:: code/benchmarks/rastrigin.py
       :width: 67 %
    """
    return (10 * len(individual) + sum((gene * gene - 10 * cos(2 * pi * gene) for gene in individual)),)