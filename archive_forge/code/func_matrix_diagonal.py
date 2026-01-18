from ..pari import pari
import fractions
def matrix_diagonal(m):
    return [r[i] for i, r in enumerate(m)]