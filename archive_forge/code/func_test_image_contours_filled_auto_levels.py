import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_contours_filled_auto_levels(self):
    z = np.array([[0, 1, 0], [3, 4, 5], [6, 7, 8]])
    img = Image(z)
    for nlevels in range(3, 20):
        op_contours = contours(img, filled=True, levels=nlevels)
        levels = [item['z'] for item in op_contours.data]
        assert len(levels) <= nlevels + 1
        delta = 0.5 * (levels[1] - levels[0])
        assert np.min(levels) <= z.min() + delta
        assert np.max(levels) >= z.max() - delta