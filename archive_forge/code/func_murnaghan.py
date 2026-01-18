import warnings
from ase.units import kJ
import numpy as np
from scipy.optimize import curve_fit
def murnaghan(V, E0, B0, BP, V0):
    """From PRB 28,5480 (1983"""
    E = E0 + B0 * V / BP * ((V0 / V) ** BP / (BP - 1) + 1) - V0 * B0 / (BP - 1)
    return E