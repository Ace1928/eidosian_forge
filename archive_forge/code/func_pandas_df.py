from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
@pytest.fixture(scope='module')
def pandas_df(self):
    return pd.DataFrame({'x': [-1, 0, 1, -1, 0, 1, -1, 0, 1], 'y': [1, 1, 1, 0, 0, 0, -1, -1, -1]}, dtype=float)