from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
def matrix1(self):
    from sage.all import RIF, CIF, matrix
    return matrix([[CIF(RIF(1.3), RIF(-0.4)), CIF(RIF(5.6), RIF(2.3))], [CIF(RIF(-0.3), RIF(0.1)), CIF(1)]])