import numpy as np
import pytest
from skimage._shared.utils import _supported_float_type
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_2d_motion(dtype):
    rng = np.random.default_rng(0)
    image0 = rng.normal(size=(256, 256))
    gt_flow, image1 = _sin_flow_gen(image0)
    image1 = image1.astype(dtype, copy=False)
    float_dtype = _supported_float_type(dtype)
    flow = optical_flow_tvl1(image0, image1, attachment=5, dtype=float_dtype)
    assert flow.dtype == float_dtype
    assert abs(flow - gt_flow).mean() < 0.5
    if dtype != float_dtype:
        with pytest.raises(ValueError):
            optical_flow_tvl1(image0, image1, attachment=5, dtype=dtype)