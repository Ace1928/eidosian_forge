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
def test_dmap_apply_not_dynamic_with_kwarg(self):
    applied = self.dmap.apply(lambda x, label: x.relabel(label), dynamic=False, label='Test')
    self.assertEqual(applied, HoloMap(self.dmap[[0, 1, 2]]).relabel('Test'))