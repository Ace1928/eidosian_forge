from ..pari import pari
import fractions
def max_abs_of_row(m, row):
    return max([abs(x) for x in m[row]])