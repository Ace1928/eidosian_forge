import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
def cohesive_potential(r):
    """
        returns the cohesive potential as a equation 28 in pair_potential
        r - numpy array with the values of distance to compute the pair function
        """
    d = 4.400224
    rho = (r - d) ** 2.0
    rho[r > d] = 0.0
    return rho