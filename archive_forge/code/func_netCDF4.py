import numpy as np
import pytest
import warnings
from ase import Atom, Atoms
from ase.io import read
from ase.io import NetCDFTrajectory
@pytest.fixture(scope='module')
def netCDF4():
    return pytest.importorskip('netCDF4')