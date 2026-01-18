import os
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from ase import units, Atoms
import ase.io
from ase.calculators.qmmm import ForceConstantCalculator
from ase.vibrations import Vibrations, VibrationsData
from ase.thermochemistry import IdealGasThermo
@pytest.fixture
def random_dimer(self):
    rng = np.random.RandomState(42)
    d = 1 + 0.5 * rng.rand()
    z_values = rng.randint(1, high=50, size=2)
    hessian = rng.rand(6, 6)
    hessian += hessian.T
    atoms = Atoms(z_values, [[0, 0, 0], [0, 0, d]])
    ref_atoms = atoms.copy()
    atoms.calc = ForceConstantCalculator(D=hessian, ref=ref_atoms, f0=np.zeros((2, 3)))
    return atoms