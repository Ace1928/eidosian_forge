import pytest
import numpy as np
from ase.build import bulk, make_supercell
from ase.lattice import FCC, BCC
from ase.calculators.emt import EMT
def test_bcc_to_cub_transformation():
    bcc = bulk('Fe', a=a)
    P = FCC(2.0).tocell()
    assert np.allclose(np.linalg.det(P), 2)
    cubatoms = make_supercell(bcc, P)
    assert np.allclose(cubatoms.cell, a * np.eye(3))