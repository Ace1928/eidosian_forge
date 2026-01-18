import numpy as np
import pytest
from ase.io.formats import ioformats, match_magic
def test_lammpsdump_order(fmt, lammpsdump):
    ref_order = np.array([1, 3, 2])
    atoms = fmt.parse_atoms(lammpsdump(have_id=True))
    assert atoms.cell.orthorhombic
    assert pytest.approx(atoms.cell.lengths()) == [4.0, 5.0, 20.0]
    assert pytest.approx(atoms.positions) == ref_positions[ref_order - 1]