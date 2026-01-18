from ..pari import pari
import fractions
def matrix_mult_vector(m, v):
    return [_inner_product(row, v) for row in m]