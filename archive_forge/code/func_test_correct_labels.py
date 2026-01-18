import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
@pytest.mark.matplotlib
@pytest.mark.parametrize('labels', [{}, {'xlabel': 'X', 'ylabel': 'Y'}])
@pytest.mark.parametrize('x_size', [30, 50])
@pytest.mark.parametrize('y_size', [30, 50])
@pytest.mark.parametrize('line', [None, '45', 's', 'r', 'q'])
def test_correct_labels(close_figures, reset_randomstate, line, x_size, y_size, labels):
    rs = np.random.RandomState(9876554)
    x = rs.normal(loc=0, scale=0.1, size=x_size)
    y = rs.standard_t(3, size=y_size)
    pp_x = sm.ProbPlot(x)
    pp_y = sm.ProbPlot(y)
    fig = qqplot_2samples(pp_x, pp_y, line=line, **labels)
    ax = fig.get_axes()[0]
    x_label = ax.get_xlabel()
    y_label = ax.get_ylabel()
    if x_size < y_size:
        if not labels:
            assert '2nd' in x_label
            assert '1st' in y_label
        else:
            assert 'Y' in x_label
            assert 'X' in y_label
    elif not labels:
        assert '1st' in x_label
        assert '2nd' in y_label
    else:
        assert 'X' in x_label
        assert 'Y' in y_label