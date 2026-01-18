import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
def pouriertarantola(V, E0, B0, BP, V0):
    """Pourier-Tarantola equation from PRB 70, 224107"""
    eta = (V / V0) ** (1 / 3)
    squiggle = -3 * np.log(eta)
    E = E0 + B0 * V0 * squiggle ** 2 / 6 * (3 + squiggle * (BP - 2))
    return E