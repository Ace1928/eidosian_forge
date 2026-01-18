import itertools
from ase import Atoms
from ase.geometry import get_distances
from ase.lattice.cubic import FaceCenteredCubic
def test_antisymmetry():
    size = 2
    atoms = FaceCenteredCubic(size=[size, size, size], symbol='Cu', latticeconstant=2, pbc=(1, 1, 1))
    vmin, vlen = get_distances(atoms.get_positions(), cell=atoms.cell, pbc=True)
    assert (vlen == vlen.T).all()
    for i, j in itertools.combinations(range(len(atoms)), 2):
        assert (vmin[i, j] == -vmin[j, i]).all()