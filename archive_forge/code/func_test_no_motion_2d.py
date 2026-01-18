import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
def test_no_motion_2d():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(256, 256))
    flow = optical_flow_tvl1(img, img)
    assert np.all(flow == 0)