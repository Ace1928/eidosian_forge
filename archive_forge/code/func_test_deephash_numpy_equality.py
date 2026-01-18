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
def test_deephash_numpy_equality(self):
    self.assertEqual(deephash(np.array([1, 2, 3])), deephash(np.array([1, 2, 3])))