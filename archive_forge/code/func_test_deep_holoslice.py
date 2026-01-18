import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_deep_holoslice(self):
    selection = self.img_map.select(a=(1, 3), b=(1, 3), x=(None, 0), y=(None, 0))
    self.assertEqual(selection, self.img_map[1:3, 1:3, :0, :0])