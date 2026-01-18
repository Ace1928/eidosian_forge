from ..pari import pari
import fractions
def matrix_modulo(m, mod):
    return [vector_modulo(row, mod) for row in m]