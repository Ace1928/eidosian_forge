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
def test_cohn(self):
    cols = ['nuncen_above', 'nobs_below', 'ncen_equal', 'prob_exceedance']
    cohn = ros.cohn_numbers(self.df, self.rescol, self.cencol)
    assert_frame_equal(np.round(cohn[cols], 3), np.round(self.expected_cohn[cols], 3))