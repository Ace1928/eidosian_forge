import pickle
import warnings
from unittest import skipIf
import numpy as np
import pandas as pd
import param
import holoviews as hv
from holoviews.core.data import Dataset
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_bin_transform(self):
    expr = dim('int').bin([0, 5, 10])
    expected = pd.Series([2.5, 2.5, 2.5, 2.5, 2.5, 7.5, 7.5, 7.5, 7.5, 7.5])
    self.assert_apply(expr, expected)