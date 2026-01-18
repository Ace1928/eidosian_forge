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
def test_censored_2(self):
    row = {'censored': True, 'det_limit_index': 4, 'rank': 2}
    result = ros._ros_plot_pos(row, 'censored', self.cohn)
    assert_equal(result, 0.41739130434782606)