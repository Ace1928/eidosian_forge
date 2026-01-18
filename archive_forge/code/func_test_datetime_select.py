import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_datetime_select(self):
    s, e = ('1999-12-31', '2000-1-2')
    curve = self.datetime_fn()
    overlay = curve * self.datetime_fn()
    for el in [curve, overlay]:
        v = el.select(time=(s, e))
        self.assertEqual(v, el[s:e])
        self.assertEqual(el.select(time=(dt.datetime(1999, 12, 31), dt.datetime(2000, 1, 2))), el[s:e])
        self.assertEqual(el.select(time=(pd.Timestamp(s), pd.Timestamp(e))), el[pd.Timestamp(s):pd.Timestamp(e)])