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
def test_isfinite_pandas_period_series(self):
    daily = pd.date_range('2017-1-1', '2017-1-3', freq='D').to_period('D').to_series()
    self.assertEqual(isfinite(daily), np.array([True, True, True]))