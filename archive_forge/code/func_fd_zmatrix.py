import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
@pytest.fixture
def fd_zmatrix():
    fd_zmatrix = StringIO(_zmatrix_file_text)
    return fd_zmatrix