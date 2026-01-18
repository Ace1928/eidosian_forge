import numpy as np
import pytest
from shapely import LineString
def test_data_destriding(self):
    coords = np.array([[12, 34], [56, 78]], dtype=np.float32)
    processed_coords = np.array(LineString(coords[::-1]).coords)
    assert coords[::-1].tolist() == processed_coords.tolist()