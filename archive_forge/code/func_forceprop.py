import pytest
import numpy as np
from ase.outputs import Properties, all_outputs
@pytest.fixture
def forceprop():
    dct = dict(forces=np.zeros((natoms, 3)))
    props = Properties(dct)
    return props