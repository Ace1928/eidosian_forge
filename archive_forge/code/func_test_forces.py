import numpy as np
from scipy.optimize import check_grad
from ase import Atoms
from ase.vibrations import Vibrations
from ase.calculators.morse import MorsePotential, fcut, fcut_d
from ase.build import bulk
def test_forces():
    atoms = bulk('Cu', cubic=True)
    atoms.calc = MorsePotential(A=4.0, epsilon=1.0, r0=2.55)
    atoms.rattle(0.1)
    forces = atoms.get_forces()
    numerical_forces = atoms.calc.calculate_numerical_forces(atoms, d=1e-05)
    assert np.abs(forces - numerical_forces).max() < 1e-05