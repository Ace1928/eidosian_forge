import numpy as np
from skimage.transform import frt2, ifrt2
def test_frt():
    SIZE = 59
    L = np.tri(SIZE, dtype=np.int32) + np.tri(SIZE, dtype=np.int32)[::-1]
    f = frt2(L)
    fi = ifrt2(f)
    assert np.array_equal(L, fi)