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
def test_no_streams(self):
    result = wrap_tuple_streams((1, 2), [], [])
    self.assertEqual(result, (1, 2))