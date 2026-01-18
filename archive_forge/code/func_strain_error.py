import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
def strain_error(at0, u_ref, u, cutoff, mask):
    I, J = neighbor_list('ij', at0, cutoff)
    I, J = np.array([(i, j) for i, j in zip(I, J) if mask[i]]).T
    v = u_ref - u
    dv = np.linalg.norm(v[I, :] - v[J, :], axis=1)
    return np.linalg.norm(dv)