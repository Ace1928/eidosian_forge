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
def fd_incorrect_zmatrix_var():
    incorrect_zmatrix_text = ''
    for i, line in enumerate(_zmatrix_file_text.split('\n')):
        if i == 10:
            incorrect_zmatrix_text += 'H 1 test 2 a1 \n'
        elif i == 18:
            incorrect_zmatrix_text += 'Constants: \n'
        else:
            incorrect_zmatrix_text += line + '\n'
    return StringIO(incorrect_zmatrix_text)