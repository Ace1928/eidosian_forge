import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_bivariate_dict_constructor(self):
    dist = Bivariate({'x': [0, 1, 2], 'y': [0, 1, 2]}, ['x', 'y'])
    assert dist.kdims == [Dimension('x'), Dimension('y')]
    assert dist.vdims == [Dimension('Density')]