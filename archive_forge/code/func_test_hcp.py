import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
def test_hcp():
    atoms = HexagonalClosedPacked(symbol='Mg', directions=[[1, -1, 0, 0], [1, 0, -1, 0], [0, 0, 0, 1]])
    print(atoms.get_cell())