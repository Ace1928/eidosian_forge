import pytest
from ase.calculators.calculator import kpts2kpts
from ase.lattice import all_variants
from ase import Atoms
@pytest.mark.parametrize('lat', all_variants())
def test_kpts2kpts(lat):
    print()
    print(lat)
    bandpath = lat.bandpath()
    a = Atoms()
    a.cell = lat.tocell().complete()
    a.pbc[:lat.ndim] = True
    path = {'path': bandpath.path}
    bandpath2 = kpts2kpts(path, atoms=a)
    print('cell', a.cell)
    print('Original', bandpath)
    print('path', path)
    print('Produced by kpts2kpts', bandpath2)
    sp = set(bandpath.special_points)
    sp2 = set(bandpath2.special_points)
    msg = 'Input and output bandpath from kpts2kpts dont agree!\nInput: {}\n Output: {}'.format(bandpath, bandpath2)
    assert sp == sp2, msg