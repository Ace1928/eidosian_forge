import datetime
import math
import unittest
from itertools import product
import numpy as np
import pandas as pd
from holoviews import Dimension, Element
from holoviews.core.util import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PointerXY
def test_date_range_1_hour(self):
    start = np.datetime64(datetime.datetime(2017, 1, 1))
    end = start + np.timedelta64(1, 'h')
    drange = date_range(start, end, 6)
    self.assertEqual(drange[0], start + np.timedelta64(5, 'm'))
    self.assertEqual(drange[-1], end - np.timedelta64(5, 'm'))