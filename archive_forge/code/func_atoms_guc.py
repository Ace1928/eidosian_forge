from math import pi, sqrt, cos
import pytest
import numpy as np
from ase import Atoms
from ase import data
from ase.lattice.cubic import FaceCenteredCubic
@pytest.fixture
def atoms_guc():
    return FaceCenteredCubic(size=(5, 5, 5), directions=[[1, 0, 0], [0, 1, 0], [1, 0, 1]], symbol=symb, pbc=(1, 1, 0))