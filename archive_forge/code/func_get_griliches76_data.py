from statsmodels.compat.python import lrange, lmap
import os
import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
import statsmodels.sandbox.regression.gmm as gmm
def get_griliches76_data():
    curdir = os.path.split(__file__)[0]
    path = os.path.join(curdir, 'griliches76.dta')
    griliches76_data = pd.read_stata(path)
    years = griliches76_data['year'].unique()
    N = griliches76_data.shape[0]
    for yr in years:
        griliches76_data['D_%i' % yr] = np.zeros(N)
        for i in range(N):
            if griliches76_data.loc[griliches76_data.index[i], 'year'] == yr:
                griliches76_data.loc[griliches76_data.index[i], 'D_%i' % yr] = 1
            else:
                pass
    griliches76_data['const'] = 1
    X = add_constant(griliches76_data[['s', 'iq', 'expr', 'tenure', 'rns', 'smsa', 'D_67', 'D_68', 'D_69', 'D_70', 'D_71', 'D_73']], prepend=True)
    Z = add_constant(griliches76_data[['expr', 'tenure', 'rns', 'smsa', 'D_67', 'D_68', 'D_69', 'D_70', 'D_71', 'D_73', 'med', 'kww', 'age', 'mrt']])
    Y = griliches76_data['lw']
    return (Y, X, Z)