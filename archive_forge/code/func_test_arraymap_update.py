import numpy as np
import pytest
from skimage.util._map_array import map_array, ArrayMap
from skimage._shared import testing
def test_arraymap_update():
    in_values = np.unique(np.random.randint(0, 200, size=5))
    out_values = np.random.random(len(in_values))
    m = ArrayMap(in_values, out_values)
    image = np.random.randint(1, len(m), size=(512, 512))
    assert np.all(m[image] < 1)
    m[1:] += 1
    assert np.all(m[image] >= 1)