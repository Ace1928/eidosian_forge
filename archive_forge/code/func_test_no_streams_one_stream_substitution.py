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
def test_no_streams_one_stream_substitution(self):
    result = wrap_tuple_streams((None, 3), [Dimension('x'), Dimension('y')], [PointerXY(x=-5, y=10)])
    self.assertEqual(result, (-5, 3))