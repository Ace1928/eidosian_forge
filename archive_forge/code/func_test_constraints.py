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
@pytest.mark.parametrize('constraint', [FixAtoms(indices=(0, 2)), FixCartesian(1, mask=(1, 0, 1)), [FixCartesian(0), FixCartesian(2)]])
def test_constraints(constraint):
    atoms = molecule('H2O')
    atoms.set_constraint(constraint)
    columns = ['symbols', 'positions', 'move_mask']
    ase.io.write('tmp.xyz', atoms, columns=columns)
    atoms2 = ase.io.read('tmp.xyz')
    assert not compare_atoms(atoms, atoms2)
    constraint2 = atoms2.constraints
    cls = type(constraint)
    if cls == FixAtoms:
        assert len(constraint2) == 1
        assert isinstance(constraint2[0], cls)
        assert np.all(constraint2[0].index == constraint.index)
    elif cls == FixCartesian:
        assert len(constraint2) == len(atoms)
        assert isinstance(constraint2[0], cls)
        assert np.all(constraint2[0].mask)
        assert np.all(constraint2[1].mask == constraint.mask)
        assert np.all(constraint2[2].mask)
    elif cls == list:
        assert len(constraint2) == len(atoms)
        assert np.all(constraint2[0].mask == constraint[0].mask)
        assert np.all(constraint2[1].mask)
        assert np.all(constraint2[2].mask == constraint[1].mask)