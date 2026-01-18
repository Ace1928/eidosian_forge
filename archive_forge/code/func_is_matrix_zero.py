from ..pari import pari
import fractions
def is_matrix_zero(m):
    for row in m:
        if not is_vector_zero(row):
            return False
    return True