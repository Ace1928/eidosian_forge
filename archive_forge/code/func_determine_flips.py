import re
import string
def determine_flips(matrices, orientable):
    """
    Returns pairs [(l,m)] for each given matrix. Multiplying the columns of
    each matrix with the respective pair brings the matrix in "canonical" form.
    """
    if orientable:
        det_sign = matrices[0][0, 0] * matrices[0][1, 1] - matrices[0][0, 1] * matrices[0][1, 0]
        return [(sgn_column(matrix, 0), sgn_column(matrix, 0) * det_sign) for matrix in matrices]
    else:
        return [(sgn_column(matrix, 0), sgn_column(matrix, 1)) for matrix in matrices]