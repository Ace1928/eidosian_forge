import os
import sys
import pytest
from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type
def test_check_atoms(atoms):
    """Test checking atoms passes for a good atoms object"""
    check_atoms(atoms)
    check_pbc(atoms)
    check_cell(atoms)