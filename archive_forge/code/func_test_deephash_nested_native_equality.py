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
def test_deephash_nested_native_equality(self):
    obj1 = [[1, 2], (3, 6, 7, [True]), 'a', 9.2, 42, {1: 3, 2: 'c'}]
    obj2 = [[1, 2], (3, 6, 7, [True]), 'a', 9.2, 42, {1: 3, 2: 'c'}]
    self.assertEqual(deephash(obj1), deephash(obj2))