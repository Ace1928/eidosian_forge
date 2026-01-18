import os
import pytest
from ase.db import connect
from ase import Atoms
from ase.calculators.emt import EMT
from ase.build import molecule
@pytest.fixture
def h2o():
    return molecule('H2O')