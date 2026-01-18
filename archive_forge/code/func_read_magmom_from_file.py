import pytest
from unittest import mock
import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool
from ase.build import bulk
def read_magmom_from_file(filename) -> np.ndarray:
    """Helper function to parse the magnetic moments from an INCAR file"""
    found = False
    with open(filename) as file:
        for line in file:
            if 'MAGMOM = ' in line:
                found = True
                parts = line.strip().split()[2:]
                new_magmom = []
                for part in parts:
                    n, val = part.split('*')
                    new_magmom += int(n) * [float(val)]
                break
    assert found
    return np.array(new_magmom)