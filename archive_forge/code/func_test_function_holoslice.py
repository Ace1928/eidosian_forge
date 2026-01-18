import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_function_holoslice(self):
    self.assertEqual(self.img_map.select(a=lambda x: 1 <= x < 3, b=lambda x: 1 <= x < 3), self.img_map[1:3, 1:3])