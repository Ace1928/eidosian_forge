from ..pari import pari
import fractions
def row_is_zero(m, row):
    if row < 0:
        return True
    return is_vector_zero(m[row])