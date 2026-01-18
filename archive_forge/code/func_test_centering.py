import numpy as np
import pytest
from ase.cluster.decahedron import Decahedron
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.neighborlist import neighbor_list
@pytest.mark.parametrize('cluster', clusters())
def test_centering(cluster):
    assert cluster.cell.rank == 0
    assert cluster.positions.sum(0) == pytest.approx(np.zeros(3), abs=1e-10)