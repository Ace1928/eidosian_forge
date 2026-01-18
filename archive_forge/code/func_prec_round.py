import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
def prec_round(a, prec=2):
    """
    To make hierarchical sorting different from non-hierarchical sorting
    with floats.
    """
    if a == 0:
        return a
    else:
        s = 1 if a > 0 else -1
        m = np.log10(s * a) // 1
        c = np.log10(s * a) % 1
    return s * np.round(10 ** c, prec) * 10 ** m