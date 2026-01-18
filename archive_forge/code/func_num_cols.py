from ..pari import pari
import fractions
def num_cols(m):
    if len(m) == 0:
        return 0
    return len(m[0])