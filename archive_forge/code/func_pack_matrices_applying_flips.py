import re
import string
def pack_matrices_applying_flips(matrices, flips):
    """
    Multiplies the columns of each matrix by the entries in flips and
    packs all the matrices into one array, column-major.
    """
    result = []
    for matrix, flip in zip(matrices, flips):
        for col in range(2):
            for row in range(2):
                result.append(matrix[row, col] * flip[col])
    return result