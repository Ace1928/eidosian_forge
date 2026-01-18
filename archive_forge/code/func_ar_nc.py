import pytest
from numpy.testing import assert_allclose
from ase.cluster.icosahedron import Icosahedron
from ase.data import atomic_numbers, atomic_masses
from ase.optimize import LBFGS
@pytest.fixture
def ar_nc():
    ar_nc = Icosahedron('Ar', noshells=2)
    ar_nc.cell = [[300, 0, 0], [0, 300, 0], [0, 0, 300]]
    ar_nc.pbc = True
    return ar_nc