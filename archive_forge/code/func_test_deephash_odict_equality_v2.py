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
def test_deephash_odict_equality_v2(self):
    odict1 = dict([(1, 'a'), (2, 'b')])
    odict2 = dict([(1, 'a'), (2, 'c')])
    self.assertNotEqual(deephash(odict1), deephash(odict2))