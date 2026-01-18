import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
def pair_potential(r):
    """
        returns the pair potential as a equation 27 in pair_potential
        r - numpy array with the values of distance to compute the pair function
        """
    c = 3.25
    c0 = 47.1346499
    c1 = -33.7665655
    c2 = 6.2541999
    energy = (c0 + c1 * r + c2 * r ** 2.0) * (r - c) ** 2.0
    energy[r > c] = 0.0
    return energy