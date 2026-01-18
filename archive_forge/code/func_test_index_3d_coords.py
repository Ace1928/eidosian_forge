import numpy as np
import pytest
from shapely import LineString
def test_index_3d_coords(self):
    c = [(float(x), float(-x), float(x * 2)) for x in range(4)]
    g = LineString(c)
    for i in range(-4, 4):
        assert g.coords[i] == c[i]
    with pytest.raises(IndexError):
        g.coords[4]
    with pytest.raises(IndexError):
        g.coords[-5]