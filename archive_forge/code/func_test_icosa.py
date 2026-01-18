import numpy as np
import pytest
from ase.cluster.decahedron import Decahedron
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.neighborlist import neighbor_list
@pytest.mark.parametrize('shells', range(1, 7))
def test_icosa(shells):
    atoms = Icosahedron(sym, shells)
    assert len(atoms) == ico_cubocta_sizes[shells]
    coordination = coordination_numbers(atoms)
    if shells == 1:
        return
    assert min(coordination) == ico_corner_coordination
    ncorners = sum(coordination == ico_corner_coordination)
    assert ncorners == ico_corners