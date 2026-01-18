import pytest
import numpy as np
from ase.cell import Cell
def test_handedness(cell):
    assert cell.handedness == 1
    cell[0] *= -1
    assert cell.handedness == -1
    cell[0] = 0
    assert cell.handedness == 0