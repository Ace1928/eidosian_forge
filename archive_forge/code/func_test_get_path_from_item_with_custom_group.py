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
def test_get_path_from_item_with_custom_group(self):
    path = get_path((('Custom',), Element('Test')))
    self.assertEqual(path, ('Custom',))