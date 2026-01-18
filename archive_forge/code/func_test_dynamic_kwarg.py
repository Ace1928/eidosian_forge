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
def test_dynamic_kwarg(self):
    p = Params(a=1)
    expr = dim('float').round(decimals=p.param.a)
    self.assertEqual(list(expr.params.values()), [p.param.a])
    self.assert_apply(expr, np.round(self.linear_floats, 1))
    p.a = 2
    self.assert_apply(expr, np.round(self.linear_floats, 2))