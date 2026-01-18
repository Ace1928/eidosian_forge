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
def test_element_apply_function_with_dependencies_non_dynamic(self):
    pinst = ParamClass()

    @param.depends(pinst.param.label)
    def get_label(label):
        return label + '!'
    applied = self.element.apply('relabel', dynamic=False, label=get_label)
    self.assertEqual(applied, self.element.relabel('Test!'))