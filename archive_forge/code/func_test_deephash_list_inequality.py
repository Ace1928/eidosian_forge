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
def test_deephash_list_inequality(self):
    obj1 = [1, 2, 3]
    obj2 = [1, 2, 3, 4]
    self.assertNotEqual(deephash(obj1), deephash(obj2))