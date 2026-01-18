import pytest
import numpy as np
from ase.dft.kpoints import resolve_custom_points
@pytest.fixture
def special_points():
    return dict(A=np.zeros(3), B=np.ones(3))