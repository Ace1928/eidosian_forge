import pytest
import numpy as np
from ase import Atoms
@pytest.fixture
def lammpsdata_file_path(datadir):
    return datadir / 'lammpsdata_input.data'