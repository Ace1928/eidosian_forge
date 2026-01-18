import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@pytest.mark.matplotlib
def test_abline_ab(self, close_figures):
    mod = self.mod
    intercept, slope = mod.params
    fig = abline_plot(intercept=intercept, slope=slope)
    close_or_save(pdf, fig)