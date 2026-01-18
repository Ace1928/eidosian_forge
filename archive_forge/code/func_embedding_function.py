import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
def embedding_function(rho):
    """
        returns energy as a function of electronic density from eq 3
        """
    A = 1.896373
    energy = -A * np.sqrt(rho)
    return energy