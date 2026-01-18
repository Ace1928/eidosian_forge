import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from ase.geometry import minkowski_reduce
from ase.cell import Cell
def test_issue():
    x = [[8.972058879514716, 0.0009788104586639142, 0.0005932485724084841], [4.485181755775297, 7.770520334862034, 0.00043663339838788054], [4.484671994095723, 2.5902066679984634, 16.25695615743613]]
    cell = Cell(x)
    cell.minkowski_reduce()