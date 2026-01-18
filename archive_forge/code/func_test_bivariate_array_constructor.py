import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_bivariate_array_constructor(self):
    dist = Bivariate(np.array([[0, 1, 2], [0, 1, 2]]))
    assert dist.kdims == [Dimension('x'), Dimension('y')]
    assert dist.vdims == [Dimension('Density')]