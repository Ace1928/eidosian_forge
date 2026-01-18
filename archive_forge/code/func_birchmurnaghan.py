import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
def birchmurnaghan(V, E0, B0, BP, V0):
    """
    BirchMurnaghan equation from PRB 70, 224107
    Eq. (3) in the paper. Note that there's a typo in the paper and it uses
    inversed expression for eta.
    """
    eta = (V0 / V) ** (1 / 3)
    E = E0 + 9 * B0 * V0 / 16 * (eta ** 2 - 1) ** 2 * (6 + BP * (eta ** 2 - 1) - 4 * eta ** 2)
    return E