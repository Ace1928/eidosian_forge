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
def test_deephash_dict_equality_v1(self):
    self.assertEqual(deephash({1: 'a', 2: 'b'}), deephash({2: 'b', 1: 'a'}))