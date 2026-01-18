import numpy as np
import pytest
from skimage.morphology import flood, flood_fill
def test_float16():
    image = np.array([9.0, 0.1, 42], dtype=np.float16)
    with pytest.raises(TypeError, match='dtype of `image` is float16'):
        flood_fill(image, 0, 1)