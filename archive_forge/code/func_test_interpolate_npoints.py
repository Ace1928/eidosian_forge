import numpy as np
import pytest
from ase.lattice import MCLC
def test_interpolate_npoints(bandpath):
    path = bandpath.interpolate(npoints=42)
    assert len(path.kpts) == 42