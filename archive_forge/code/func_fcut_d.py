import numpy as np
from ase.calculators.calculator import Calculator
from ase.neighborlist import neighbor_list
def fcut_d(r, r0, r1):
    """
    Derivative of fcut() function defined above
    """
    s = 1 - (r - r0) / (r1 - r0)
    return -(((0.0 < s) & (s < 1.0)) * ((30 * s ** 4 - 60 * s ** 3 + 30 * s ** 2) / (r1 - r0)))