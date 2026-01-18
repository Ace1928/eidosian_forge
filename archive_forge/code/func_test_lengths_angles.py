import pytest
import numpy as np
from ase.cell import Cell
def test_lengths_angles(cell):
    assert cell.cellpar() == pytest.approx(testcellpar)
    assert cell.lengths() == pytest.approx(testcellpar[:3])
    assert cell.angles() == pytest.approx(testcellpar[3:])