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
def test_arabic_five_sanitized(self):
    """
        Note: There would be a clash if you mixed the languages of
        your digits! E.g. arabic ٥ five and urdu ۵ five
        """
    try:
        sanitize_identifier('٥')
    except SyntaxError as e:
        assert str(e).startswith("String '٥' cannot be sanitized")