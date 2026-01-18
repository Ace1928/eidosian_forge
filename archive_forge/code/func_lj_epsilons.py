import pytest
import numpy as np
from ase import Atoms
@pytest.fixture
def lj_epsilons():
    return {'eps_orig': 2.5, 'eps_modified': 4.25}