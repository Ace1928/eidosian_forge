from ase import Atoms
from ase.neb import interpolate
from ase.constraints import FixAtoms
import numpy as np
import pytest
@pytest.fixture
def average_pos(initial, final):
    return np.average([initial.positions, final.positions], axis=0)