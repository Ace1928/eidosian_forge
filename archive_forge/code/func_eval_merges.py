import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
def eval_merges(merges):
    R = PolynomialRing(QQ, 'x', 49).fraction_field()
    A = matrix(7, 7, R.gens())
    for a, b in merges:
        A = strand_matrix_merge(A, a, b)
    return A