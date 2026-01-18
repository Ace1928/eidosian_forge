import time
import numpy as np
import pytest
from scipy.spatial.distance import pdist, minkowski
from skimage._shared.coord import ensure_spacing
@pytest.mark.parametrize('p', [1, 2, np.inf])
@pytest.mark.parametrize('size', [30, 50, None])
def test_ensure_spacing_p_norm(p, size):
    coord = np.random.randn(100, 2)
    spacing = np.median(pdist(coord, metric=minkowski, p=p))
    out = ensure_spacing(coord, spacing=spacing, p_norm=p, min_split_size=size)
    assert pdist(out, metric=minkowski, p=p).min() > spacing