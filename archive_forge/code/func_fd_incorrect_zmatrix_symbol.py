import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def fd_incorrect_zmatrix_symbol():
    incorrect_zmatrix_text = ''
    for i, line in enumerate(_zmatrix_file_text.split('\n')):
        if i == 10:
            incorrect_zmatrix_text += 'UnknownSymbol 0 1.31 0.00 0.00\n'
        else:
            incorrect_zmatrix_text += line + '\n'
    return StringIO(incorrect_zmatrix_text)