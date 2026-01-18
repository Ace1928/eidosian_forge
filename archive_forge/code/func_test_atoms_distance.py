import itertools
from ase import Atoms
from ase.geometry import get_distances
from ase.lattice.cubic import FaceCenteredCubic
def test_atoms_distance():
    a = Atoms('HOC', positions=[(1, 1, 1), (3, 1, 1), (6, 1, 1)])
    a.set_cell((9, 2, 2))
    a.set_pbc((True, False, False))
    assert a.get_distance(0, 1, mic=True) == 2
    assert a.get_distance(1, 2, mic=True) == 3
    assert a.get_distance(0, 2, mic=True) == 4
    assert a.get_distance(0, 1, mic=False) == 2
    assert a.get_distance(1, 2, mic=False) == 3
    assert a.get_distance(0, 2, mic=False) == 5
    assert (a.get_distances(0, [1, 2], mic=True) == [2, 4]).all()
    assert (a.get_distances(0, [1, 2], mic=False) == [2, 5]).all()
    assert (a.get_all_distances(mic=True) == [[0, 2, 4], [2, 0, 3], [4, 3, 0]]).all()
    assert (a.get_all_distances(mic=False) == [[0, 2, 5], [2, 0, 3], [5, 3, 0]]).all()
    old = a.get_distance(0, 1)
    a.set_distance(0, 1, 0.9, add=True, factor=True)
    new = a.get_distance(0, 1)
    diff = new - 0.9 * old
    assert abs(diff) < 1e-05
    old = a.get_distance(0, 1)
    a.set_distance(0, 1, 0.9, add=True)
    new = a.get_distance(0, 1)
    diff = new - old - 0.9
    assert abs(diff) < 1e-05