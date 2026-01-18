import warnings
from statsmodels.compat.pandas import PD_LT_1_4
import os
import numpy as np
import pandas as pd
from statsmodels.multivariate.factor import Factor
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
def test_example_compare_to_R_output():
    mod = Factor(X.iloc[:, 1:-1], 2, smc=False)
    results = mod.fit(tol=1e-10)
    a = np.array([[0.965392158864, 0.225880658666255], [0.967587154301, 0.212758741910989], [0.929891035996, -0.000603217967568], [0.486822656362, -0.869649573289374]])
    assert_array_almost_equal(results.loadings, a, decimal=8)
    mod = Factor(X.iloc[:, 1:-1], 2, smc=True)
    results = mod.fit()
    a = np.array([[0.97541115, 0.20280987], [0.97113975, 0.17207499], [0.9618705, -0.2004196], [0.37570708, -0.45821379]])
    assert_array_almost_equal(results.loadings, a, decimal=8)
    results.rotate('varimax')
    a = np.array([[0.98828898, -0.12587155], [0.97424206, -0.15354033], [0.84418097, -0.502714], [0.20601929, -0.55558235]])
    assert_array_almost_equal(results.loadings, a, decimal=8)
    results.rotate('quartimax')
    a = np.array([[0.98935598, 0.98242714, 0.94078972, 0.33442284], [0.117190049, 0.086943252, -0.283332952, -0.489159543]])
    assert_array_almost_equal(results.loadings, a.T, decimal=8)
    results.rotate('equamax')
    results.rotate('promax')
    results.rotate('biquartimin')
    results.rotate('oblimin')
    a = np.array([[1.0283417017, 1.00178840104, 0.71824931384, -0.00013510048], [0.06563421, 0.03096076, -0.39658839, -0.59261944]])
    assert_array_almost_equal(results.loadings, a.T, decimal=8)
    results.rotate('varimax')
    desired = '   Factor analysis results\n=============================\n      Eigenvalues\n-----------------------------\n Basal   Occ    Max      id\n-----------------------------\n 2.9609 0.3209 0.0000 -0.0000\n-----------------------------\n\n-----------------------------\n      Communality\n-----------------------------\n  Basal   Occ    Max     id\n-----------------------------\n  0.9926 0.9727 0.9654 0.3511\n-----------------------------\n\n-----------------------------\n   Pre-rotated loadings\n-----------------------------------\n            factor 0       factor 1\n-----------------------------------\nBasal         0.9754         0.2028\nOcc           0.9711         0.1721\nMax           0.9619        -0.2004\nid            0.3757        -0.4582\n-----------------------------\n\n-----------------------------\n   varimax rotated loadings\n-----------------------------------\n            factor 0       factor 1\n-----------------------------------\nBasal         0.9883        -0.1259\nOcc           0.9742        -0.1535\nMax           0.8442        -0.5027\nid            0.2060        -0.5556\n=============================\n'
    actual = results.summary().as_text()
    actual = '\n'.join((line.rstrip() for line in actual.splitlines())) + '\n'
    assert_equal(actual, desired)