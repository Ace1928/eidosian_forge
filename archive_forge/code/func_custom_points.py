import numpy as np
import pytest
from ase.lattice import MCLC
@pytest.fixture
def custom_points():
    rng = np.random.RandomState(0)
    dct = {}
    for name in ['K', 'K1', 'Kpoint', 'Kpoint1']:
        dct[name] = rng.rand(3)
    return dct