import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
@pytest.mark.matplotlib
def test_fit_obs(self, close_figures):
    df = gendat()
    imp_data = mice.MICEData(df)
    imp_data.update_all()
    plt.clf()
    for plot_points in (False, True):
        fig = imp_data.plot_fit_obs('x4', plot_points=plot_points)
        fig.get_axes()[0].set_title('plot_fit_scatterplot')
        close_or_save(pdf, fig)
        close_figures()