import numpy as np
import pytest
from ase import Atoms
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.bond_polarizability import LippincottStuttman, Linearized
def check_symmetry(alpha):
    alpha_diag = np.diagonal(alpha)
    assert alpha == pytest.approx(np.diag(alpha_diag))
    assert alpha_diag[0] == alpha_diag[1]