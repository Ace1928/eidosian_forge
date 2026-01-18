import numpy as np
import pytest
from ase.io.formats import ioformats, match_magic
@pytest.mark.parametrize('bounds,expected', bounds_parameters)
def test_lammpsdump_bounds(fmt, lammpsdump, bounds, expected):
    atoms = fmt.parse_atoms(lammpsdump(bounds=bounds))
    assert pytest.approx(atoms.cell.lengths()) == [4.0, 5.0, 20.0]
    assert np.all(atoms.get_pbc() == expected)