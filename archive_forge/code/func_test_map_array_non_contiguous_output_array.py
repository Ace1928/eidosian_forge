import numpy as np
import pytest
from skimage.util._map_array import map_array, ArrayMap
from skimage._shared import testing
def test_map_array_non_contiguous_output_array():
    labels = np.random.randint(0, 5, size=(24, 25))
    out = np.empty((24 * 3, 25 * 2))[::3, ::2]
    in_values = np.unique(labels)
    out_values = np.random.random(in_values.shape).astype(out.dtype)
    with testing.raises(ValueError):
        map_array(labels, in_values, out_values, out=out)