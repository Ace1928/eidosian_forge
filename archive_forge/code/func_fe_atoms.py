import pytest
import numpy as np
from ase.build import bulk, molecule
from ase.units import Hartree
@pytest.fixture
def fe_atoms(abinit_factory):
    atoms = bulk('Fe')
    atoms.set_initial_magnetic_moments([1])
    calc = abinit_factory.calc(nbands=8, kpts=[2, 2, 2])
    atoms.calc = calc
    return atoms