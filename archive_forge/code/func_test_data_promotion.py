import numpy as np
import pytest
from shapely import LineString
def test_data_promotion(self):
    coords = np.array([[12, 34], [56, 78]], dtype=np.float32)
    processed_coords = np.array(LineString(coords).coords)
    assert coords.tolist() == processed_coords.tolist()