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
def test_max_range2(self):
    lower, upper = max_range(self.ranges2)
    self.assertTrue(math.isnan(lower))
    self.assertTrue(math.isnan(upper))