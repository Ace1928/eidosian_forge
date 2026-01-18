from pathlib import Path
import numpy as np
import pytest
import ase.io
from ase.io import extxyz
from ase.atoms import Atoms
from ase.build import bulk
from ase.io.extxyz import escape
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixCartesian
from ase.stress import full_3x3_to_voigt_6_stress
from ase.build import molecule
def test_array_shape(at):
    at.info['bad-info'] = [[1, np.array([0, 1])], [2, np.array([0, 1])]]
    with pytest.warns(UserWarning):
        ase.io.write('to.xyz', at, format='extxyz')
    del at.info['bad-info']
    at.arrays['ns_extra_data'] = np.zeros((len(at), 1))
    assert at.arrays['ns_extra_data'].shape == (2, 1)
    ase.io.write('to_new.xyz', at, format='extxyz')
    at_new = ase.io.read('to_new.xyz')
    assert at_new.arrays['ns_extra_data'].shape == (2,)