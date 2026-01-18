import numpy as np
import pytest
from ase.lattice import MCLC
@pytest.fixture
def bandpath(lat):
    return lat.bandpath(npoints=0)