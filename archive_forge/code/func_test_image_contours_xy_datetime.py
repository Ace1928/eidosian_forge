import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_contours_xy_datetime(self):
    x = np.array(['2023-09-01', '2023-09-03', '2023-09-05'], dtype='datetime64')
    y = np.array(['2023-10-07', '2023-10-08'], dtype='datetime64')
    z = np.array([[0, 1, 0], [0, 1, 0]])
    img = Image((x, y, z))
    op_contours = contours(img, levels=[0.5])
    tz = dt.timezone.utc
    expected_x = np.array([dt.datetime(2023, 9, 2, tzinfo=tz), dt.datetime(2023, 9, 2, tzinfo=tz), np.nan, dt.datetime(2023, 9, 4, tzinfo=tz), dt.datetime(2023, 9, 4, tzinfo=tz)], dtype=object)
    expected_y = np.array([dt.datetime(2023, 10, 8, tzinfo=tz), dt.datetime(2023, 10, 7, tzinfo=tz), np.nan, dt.datetime(2023, 10, 7, tzinfo=tz), dt.datetime(2023, 10, 8, tzinfo=tz)], dtype=object)
    x = op_contours.dimension_values('x')
    mask = np.array([True, True, False, True, True])
    np.testing.assert_array_equal(x[mask], expected_x[mask])
    np.testing.assert_array_equal(x[~mask].astype(float), expected_x[~mask].astype(float))
    y = op_contours.dimension_values('y')
    np.testing.assert_array_equal(y[mask], expected_y[mask])
    np.testing.assert_array_equal(y[~mask].astype(float), expected_y[~mask].astype(float))
    np.testing.assert_array_almost_equal(op_contours.dimension_values('z'), [0.5] * 5)