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
def test_categorize_transform_dict(self):
    expr = dim('categories').categorize({'A': 'circle', 'B': 'square', 'C': 'triangle'})
    expected = pd.Series(['circle', 'square', 'triangle'] * 3 + ['circle'])
    self.assert_apply(expr, expected)