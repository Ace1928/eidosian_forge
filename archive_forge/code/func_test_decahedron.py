import numpy as np
import pytest
from ase.cluster.decahedron import Decahedron
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.neighborlist import neighbor_list
def test_decahedron():
    p = 3
    q = 4
    r = 2
    deca = Decahedron(sym, p, q, r)
    assert len(deca) == 520
    coordination = coordination_numbers(deca)
    internal_atoms = sum(coordination == fcc_maxcoordination)
    next_smaller_deca = Decahedron(sym, p - 1, q - 1, r)
    assert internal_atoms == len(next_smaller_deca)