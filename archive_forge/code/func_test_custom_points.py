import numpy as np
import pytest
from ase.lattice import MCLC
def test_custom_points(cell, custom_points):
    npoints = 42
    path = cell.bandpath('KK1,KpointKpoint1', special_points=custom_points, npoints=npoints)
    print(path)
    assert len(path.kpts) == npoints
    assert path.kpts[0] == pytest.approx(custom_points['K'])
    assert path.kpts[-1] == pytest.approx(custom_points['Kpoint1'])