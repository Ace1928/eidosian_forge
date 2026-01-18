import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
def test_hcp_cell_linearly_dependent():
    with pytest.raises(ValueError):
        HexagonalClosedPacked(symbol='Mg', directions=[[1, -1, 0, 0], [1, 0, -1, 0], [0, 1, -1, 0]])