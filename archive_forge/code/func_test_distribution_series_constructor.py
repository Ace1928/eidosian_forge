import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_distribution_series_constructor(self):
    dist = Distribution(pd.Series([0, 1, 2], name='Value'))
    assert dist.kdims == [Dimension('Value')]
    assert dist.vdims == [Dimension('Density')]