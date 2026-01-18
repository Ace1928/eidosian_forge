import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_deep_holooverlay_slice(self):
    map_slc = self.overlay_map[1:3, 1:3]
    img_slc = map_slc.map(lambda x: x[0:0.5, 0:0.5], [Image, Contours])
    selection = self.overlay_map.select(a=(1, 3), b=(1, 3), x=(0, 0.5), y=(0, 0.5))
    self.assertEqual(selection, img_slc)