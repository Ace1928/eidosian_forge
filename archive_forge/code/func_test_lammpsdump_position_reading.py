import numpy as np
import pytest
from ase.io.formats import ioformats, match_magic
@pytest.mark.parametrize('cols,scaled', [('xs ys zs', True), ('xsu ysu zsu', True), ('x y z', False), ('xu yu zu', False)])
def test_lammpsdump_position_reading(fmt, lammpsdump, cols, scaled):
    atoms = fmt.parse_atoms(lammpsdump(position_cols=cols))
    assert atoms.cell.orthorhombic
    assert pytest.approx(atoms.cell.lengths()) == [4.0, 5.0, 20.0]
    if scaled:
        assert pytest.approx(atoms.positions) == ref_positions * np.array([4, 5, 20]).T
    else:
        assert pytest.approx(atoms.positions) == ref_positions