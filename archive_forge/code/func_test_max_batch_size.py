import time
import numpy as np
import pytest
from scipy.spatial.distance import pdist, minkowski
from skimage._shared.coord import ensure_spacing
def test_max_batch_size():
    """Small batches are slow, large batches -> large allocations -> also slow.

    https://github.com/scikit-image/scikit-image/pull/6035#discussion_r751518691
    """
    coords = np.random.randint(low=0, high=1848, size=(40000, 2))
    tstart = time.time()
    ensure_spacing(coords, spacing=100, min_split_size=50, max_split_size=2000)
    dur1 = time.time() - tstart
    tstart = time.time()
    ensure_spacing(coords, spacing=100, min_split_size=50, max_split_size=20000)
    dur2 = time.time() - tstart
    assert dur1 < 1.33 * dur2