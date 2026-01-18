from math import pi
import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, kpts2ndarray
from ase.units import Bohr, Ha
def gradient_test(atoms, indices=None):
    """
    Use numeric_force to compare analytical and numerical forces on atoms

    If indices is None, test is done on all atoms.
    """
    if indices is None:
        indices = range(len(atoms))
    f = atoms.get_forces()[indices]
    print('{0:>16} {1:>20}'.format('eps', 'max(abs(df))'))
    for eps in np.logspace(-1, -8, 8):
        fn = np.zeros((len(indices), 3))
        for idx, i in enumerate(indices):
            for j in range(3):
                fn[idx, j] = numeric_force(atoms, i, j, eps)
        print('{0:16.12f} {1:20.12f}'.format(eps, abs(fn - f).max()))
    return (f, fn)