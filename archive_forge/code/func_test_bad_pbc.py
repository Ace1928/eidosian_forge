import os
import sys
import pytest
from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type
@pytest.mark.parametrize('pbc', [3 * [False], [True, False, True], [False, True, False]])
def test_bad_pbc(atoms, pbc):
    """Test handling of PBC"""
    atoms.pbc = pbc
    check_cell(atoms)
    with pytest.raises(CalculatorSetupError):
        check_pbc(atoms)
    with pytest.raises(CalculatorSetupError):
        check_atoms(atoms)
    calc = Vasp()
    atoms.calc = calc
    with pytest.raises(CalculatorSetupError):
        atoms.get_potential_energy()