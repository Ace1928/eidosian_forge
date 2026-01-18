import os
import tempfile
from unittest import SkipTest
from collections import OrderedDict
import numpy as np
from holoviews import Store
from holoviews.element import RGB, Image
from holoviews.element.comparison import ComparisonTestCase
def test_dataarray_label_precedence(self):
    plot = self.da_rgb.sel(band=1).rename('a').hvplot.image(label='b')
    assert plot.vdims[0].name == 'a'
    plot = self.da_rgb.sel(band=1).hvplot.image(label='b', value_label='c')
    assert plot.vdims[0].name == 'b'