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
def test_dmap_apply_param_method_with_dependencies(self):
    pinst = ParamClass()
    applied = self.dmap.apply(pinst.apply_label)
    self.assertEqual(len(applied.streams), 1)
    stream = applied.streams[0]
    self.assertIsInstance(stream, ParamMethod)
    self.assertEqual(stream.parameterized, pinst)
    self.assertEqual(stream.parameters, [pinst.param.label])
    self.assertEqual(applied[1], self.dmap[1].relabel('Test'))
    pinst.label = 'Another label'
    self.assertEqual(applied[1], self.dmap[1].relabel('Another label'))