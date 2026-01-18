from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
def test_equivalence_from_pairs(self):
    np.random.seed(3424)
    endog = np.random.normal(size=50)
    exog = np.random.normal(size=(50, 2))
    exog[:, 0] = 1
    groups = np.kron(np.arange(5), np.ones(10))
    groups[30:] = 3
    labels = np.kron(np.arange(5), np.ones(10)).astype(np.int32)
    labels = labels[np.random.permutation(len(labels))]
    eq = cov_struct.Equivalence(labels=labels, return_cov=True)
    model1 = gee.GEE(endog, exog, groups, cov_struct=eq)
    eq._pairs_from_labels()
    for g in model1.group_labels:
        p = eq.pairs[g]
        vl = [len(x[0]) for x in p.values()]
        m = sum(groups == g)
        assert_allclose(sum(vl), m * (m + 1) / 2)
    ixs = set()
    for g in model1.group_labels:
        for v in eq.pairs[g].values():
            for a, b in zip(v[0], v[1]):
                ky = (a, b)
                assert ky not in ixs
                ixs.add(ky)
    eq = cov_struct.Equivalence(labels=labels, return_cov=True)
    model1 = gee.GEE(endog, exog, groups, cov_struct=eq)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model1.fit(maxiter=2)