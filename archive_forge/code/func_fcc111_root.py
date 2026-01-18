from math import log10, atan2, cos, sin
from ase.build import hcp0001, fcc111, bcc111
import numpy as np
def fcc111_root(symbol, root, size, a=None, vacuum=None, orthogonal=False):
    """FCC(111) surface maniupulated to have a x unit side length
    of *root* before repeating. This also results in *root* number
    of repetitions of the cell.

    The first 20 valid roots for nonorthogonal are...
    1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25, 27,
    28, 31, 36, 37, 39, 43, 48, 49"""
    atoms = fcc111(symbol=symbol, size=(1, 1, size[2]), a=a, vacuum=vacuum, orthogonal=orthogonal)
    atoms = root_surface(atoms, root)
    atoms *= (size[0], size[1], 1)
    return atoms