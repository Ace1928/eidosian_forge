import logging
import numpy as np
import pandas as pd
from param import get_logger
from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
def test_mixed_dims_raises(self):
    arrays = [{'x': range(10), 'y' if j else 'z': range(10)} for i in range(2) for j in range(2)]
    with self.assertRaises(ValueError):
        Path(arrays, kdims=['x', 'y'], datatype=[self.datatype])