from statsmodels.compat.pandas import assert_series_equal, assert_frame_equal
from io import StringIO
from textwrap import dedent
import numpy as np
import numpy.testing as npt
import numpy
from numpy.testing import assert_equal
import pandas
import pytest
from statsmodels.imputation import ros
def test__norm_plot_pos():
    result = ros._norm_plot_pos([1, 2, 3, 4])
    expected = numpy.array([0.159104, 0.385452, 0.614548, 0.840896])
    npt.assert_array_almost_equal(result, expected)