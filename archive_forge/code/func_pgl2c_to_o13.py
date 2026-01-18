from ..matrix import matrix
def pgl2c_to_o13(m):
    """
    Converts matrix in PGL(2,C) to O13.

    Python implementation of Moebius_to_O31 in matrix_conversion.c.
    """
    return psl2c_to_o13(m / m.det().sqrt())