import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_dataarray_unnamed_value_label(self):
    plot = self.da_rgb.sel(band=1).hvplot.image(value_label='test')
    assert plot.vdims[0].name == 'test'