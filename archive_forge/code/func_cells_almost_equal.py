import pytest
import numpy as np
from ase import io
from ase.optimize import BFGS
from ase.build import bulk
def cells_almost_equal(cellA, cellB, tol=0.01):
    return (np.abs(cellA - cellB) < tol).all()