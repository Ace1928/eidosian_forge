import numpy as np
import pytest
from ase.lattice import MCLC
def test_interpolate_density(bandpath):
    path1 = bandpath.interpolate(density=10)
    path2 = bandpath.interpolate(density=20)
    assert len(path1.kpts) == len(path2.kpts) // 2