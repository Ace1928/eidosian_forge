import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def test_incorrect_mol_spec(fd_incorrect_zmatrix_var):
    """ Tests that incorrect lines in the molecule
    specification fail to be read."""
    freeze_code_line = 'H 1 1.0 2.0 3.0'
    symbol, pos = _get_atoms_info(freeze_code_line)
    with pytest.raises(ParseError):
        _get_cartesian_atom_coords(symbol, pos)
    incorrect_zmatrix = 'C4 O1 0.8 C2 121.4 O2 150.0 1'
    with pytest.raises(ParseError):
        _get_zmatrix_line(incorrect_zmatrix)
    incorrect_symbol = 'C1-7 0 1 2 3'
    with pytest.raises(ParseError):
        _validate_symbol_string(incorrect_symbol)
    with pytest.warns(UserWarning):
        with pytest.raises(ParseError):
            read_gaussian_in(fd_incorrect_zmatrix_var, True)