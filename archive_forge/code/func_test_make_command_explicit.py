import os
import sys
import pytest
from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type
def test_make_command_explicit(monkeypatch):
    """Test explicitly passing a command to the calculator"""
    for envvar in Vasp.env_commands:
        monkeypatch.setenv(envvar, 'something')
    calc = Vasp()
    my_cmd = 'my command'
    cmd = calc.make_command(my_cmd)
    assert cmd == my_cmd