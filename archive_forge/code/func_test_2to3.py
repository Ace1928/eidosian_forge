import numpy as np
import pytest
from ase import Atoms
from ase.calculators.bond_polarizability import BondPolarizability
from ase.calculators.bond_polarizability import LippincottStuttman, Linearized
def test_2to3():
    """Compare polarizabilties of one and two bonds"""
    Si2 = Atoms('Si2', positions=[[0, 0, 0], [0, 0, 2.5]])
    Si3 = Atoms('Si3', positions=[[0, 0, -2.5], [0, 0, 0], [0, 0, 2.5]])
    bp = BondPolarizability()
    bp2 = bp(Si2)
    assert bp2.shape == (3, 3)
    assert bp(Si3) == pytest.approx(2 * bp2)