import pytest
import ase.build
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.geometry.dimensionality import (analyze_dimensionality,
@pytest.mark.parametrize('method', ['TSA', 'RDA'])
def test_mx2(method):
    atoms = ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19)
    atoms.cell[2, 2] = 7
    atoms.set_pbc((1, 1, 1))
    atoms *= 2
    intervals = analyze_dimensionality(atoms, method=method)
    m = intervals[0]
    assert m.dimtype == '2D'
    assert intervals[0].dimtype == '2D'
    assert intervals[0].h == (0, 0, 2, 0)
    assert intervals[1].dimtype == '3D'
    assert intervals[1].h == (0, 0, 0, 1)
    assert intervals[2].dimtype == '0D'
    assert intervals[2].h == (24, 0, 0, 0)