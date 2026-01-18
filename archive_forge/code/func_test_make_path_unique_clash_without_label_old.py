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
def test_make_path_unique_clash_without_label_old(self):
    path = ('Element',)
    new_path = make_path_unique(path, {path: 1}, False)
    self.assertEqual(new_path, path + ('I',))