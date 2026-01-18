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
def test_dynamic_mul(self):
    p = Params(a=1)
    expr = dim('float') * p.param.a
    self.assertEqual(list(expr.params.values()), [p.param.a])
    self.assert_apply(expr, self.linear_floats)
    p.a = 2
    self.assert_apply(expr, self.linear_floats * 2)