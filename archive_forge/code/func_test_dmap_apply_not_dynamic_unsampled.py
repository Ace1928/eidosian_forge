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
def test_dmap_apply_not_dynamic_unsampled(self):
    with self.assertRaises(ValueError):
        self.dmap_unsampled.apply(lambda x: x.relabel('Test'), dynamic=False)