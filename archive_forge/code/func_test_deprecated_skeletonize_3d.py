import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage import io, draw
from skimage.data import binary_blobs
from skimage.morphology import skeletonize, skeletonize_3d
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_, parametrize, fetch
def test_deprecated_skeletonize_3d():
    image = np.ones((10, 10), dtype=bool)
    regex = 'Use `skimage\\.morphology\\.skeletonize'
    with pytest.warns(FutureWarning, match=regex) as record:
        skeletonize_3d(image)
    assert len(record) == 1
    assert record[0].filename == __file__, 'warning points at wrong file'