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
def test_deephash_dataframe_column_inequality(self):
    self.assertNotEqual(deephash(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})), deephash(pd.DataFrame({'a': [1, 2, 3], 'c': [4, 5, 6]})))