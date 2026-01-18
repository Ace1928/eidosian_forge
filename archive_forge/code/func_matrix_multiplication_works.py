from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
def matrix_multiplication_works(self, matrices):
    from sage.all import RIF, CIF, prod
    a = FinitePoint(CIF(RIF(3.5), RIF(-3.0)), RIF(8.5))
    a0 = a.translate_PGL(prod(matrices))
    for m in matrices[::-1]:
        a = a.translate_PGL(m)
    if not a.dist(a0) < RIF(1e-06):
        raise Exception('Distance %r' % a.dist(a0))