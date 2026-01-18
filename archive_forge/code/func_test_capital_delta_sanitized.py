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
def test_capital_delta_sanitized(self):
    sanitized = sanitize_identifier('Δ')
    self.assertEqual(sanitized, 'Δ')