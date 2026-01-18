import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
def test_bulk_stress():
    atoms = bulk('Ar', cubic=True)
    atoms.set_cell(atoms.cell * stretch, scale_atoms=True)
    calc = LennardJones(rc=10)
    atoms.calc = calc
    stress = atoms.get_stress()
    stresses = atoms.get_stresses()
    assert np.allclose(stress, stresses.sum(axis=0))
    pressure = sum(stress[:3]) / 3
    assert pressure == reference_pressure