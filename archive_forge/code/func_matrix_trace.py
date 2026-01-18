from ..pari import pari
import fractions
def matrix_trace(m):
    return sum(matrix_diagonal(m))