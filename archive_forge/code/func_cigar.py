import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def cigar(individual):
    """Cigar test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - none
       * - Global optima
         - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = x_0^2 + 10^6\\sum_{i=1}^N\\,x_i^2`
    """
    return (individual[0] ** 2 + 1000000.0 * sum((gene * gene for gene in individual[1:])),)