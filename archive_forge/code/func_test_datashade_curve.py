from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_datashade_curve(self):
    rgb = dynspread(datashade(self.ds.to(Curve, 'a', 'b', groupby=[]), dynamic=False), dynamic=False)
    rgb2 = dynspread(datashade(self.ds2.to(Curve, 'a', 'b', groupby=[]), dynamic=False), dynamic=False)
    self.assertNotEqual(rgb, rgb2)
    self.assertEqual(rgb.dataset, self.ds)
    ops = rgb.pipeline.operations
    self.assertEqual(len(ops), 4)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertIs(ops[1].output_type, Curve)
    self.assertIsInstance(ops[2], datashade)
    self.assertIsInstance(ops[3], dynspread)
    self.assertEqual(rgb.pipeline(rgb.dataset), rgb)
    self.assertEqual(rgb.pipeline(self.ds2), rgb2)