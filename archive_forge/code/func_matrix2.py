from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
def matrix2(self):
    from sage.all import RIF, CIF, matrix
    return matrix([[CIF(RIF(0.3), RIF(-1.4)), CIF(RIF(3.6), RIF(6.3))], [CIF(RIF(-0.3), RIF(1.1)), CIF(1)]])