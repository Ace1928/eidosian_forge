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
def test_dimension_range_date_hard_range(self):
    drange = dimension_range(self.date_range2[0], self.date_range2[1], self.date_range, (None, None))
    self.assertEqual(drange, self.date_range)