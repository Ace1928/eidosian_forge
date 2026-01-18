import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def schaffer(individual):
    """Schaffer test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \\in [-100, 100]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
       * - Function
         -  :math:`f(\\mathbf{x}) = \\sum_{i=1}^{N-1} (x_i^2+x_{i+1}^2)^{0.25} \\cdot \\
                  \\left[ \\sin^2(50\\cdot(x_i^2+x_{i+1}^2)^{0.10}) + 1.0 \\
                  \\right]`

    .. plot:: code/benchmarks/schaffer.py
        :width: 67 %
    """
    return (sum(((x ** 2 + x1 ** 2) ** 0.25 * (sin(50 * (x ** 2 + x1 ** 2) ** 0.1) ** 2 + 1.0) for x, x1 in zip(individual[:-1], individual[1:]))),)