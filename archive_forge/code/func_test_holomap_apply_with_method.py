import numpy as np
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider, RadioButtonGroup, TextInput
from holoviews import Dataset, util
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import Curve, Image
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import ParamMethod, Params
def test_holomap_apply_with_method(self):
    hmap = HoloMap({i: Image(np.array([[i, 2], [3, 4]])) for i in range(3)})
    reduced = hmap.apply.reduce(x=np.min)
    expected = HoloMap({i: Curve([(-0.25, 3), (0.25, i)], 'y', 'z') for i in range(3)})
    self.assertEqual(reduced, expected)