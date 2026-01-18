import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_image_ellipsis_slice_value_missing(self):
    data = np.random.rand(10, 10)
    try:
        hv.Image(data)[..., 'Non-existent']
    except Exception as e:
        if str(e) != "'Non-existent' is not an available value dimension":
            raise AssertionError('Unexpected exception.')