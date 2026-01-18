import numpy as np
import statsmodels.api as sm
import os
from statsmodels.stats.mediation import Mediation
import pandas as pd
from numpy.testing import assert_allclose
import patsy
import pytest
def t_est_mixedlm():
    np.random.seed(3424)
    mn = np.random.randn(5)
    c = 0.0001 * (np.random.rand(5, 5) - 0.5)
    cov = np.eye(5) + c + c.T
    rvs = np.random.multivariate_normal(mn, cov)
    rvs1 = [0.3357151, 1.26183927, 1.22539916, 0.85838887, -0.0493799]
    assert_allclose(rvs, rvs1, atol=1e-07)
    np.random.seed(3424)
    n = 200
    x = np.random.normal(size=n)
    xv = np.outer(x, np.ones(3))
    mx = np.asarray([4.0, 4, 1])
    mx /= np.sqrt(np.sum(mx ** 2))
    med = mx[0] * np.outer(x, np.ones(3))
    med += mx[1] * np.outer(np.random.normal(size=n), np.ones(3))
    med += mx[2] * np.random.normal(size=(n, 3))
    ey = np.outer(x, np.r_[0, 0.5, 1]) + med
    ex = np.asarray([5.0, 2, 2])
    ex /= np.sqrt(np.sum(ex ** 2))
    e = ex[0] * np.outer(np.random.normal(size=n), np.ones(3))
    e += ex[1] * np.outer(np.random.normal(size=n), np.r_[-1, 0, 1])
    e += ex[2] * np.random.normal(size=(n, 3))
    y = ey + e
    idx = np.outer(np.arange(n), np.ones(3))
    tim = np.outer(np.ones(n), np.r_[-1, 0, 1])
    df = pd.DataFrame({'y': y.flatten(), 'x': xv.flatten(), 'id': idx.flatten(), 'time': tim.flatten(), 'med': med.flatten()})
    dmean = [-0.13643661, -0.14266871, 99.5, 0.0, -0.15102166]
    assert_allclose(np.asarray(df.mean()), dmean, atol=1e-07)
    mediator_model = sm.MixedLM.from_formula('med ~ x', groups='id', data=df)
    outcome_model = sm.MixedLM.from_formula('y ~ med + x', groups='id', data=df)
    me = Mediation(outcome_model, mediator_model, 'x', 'med')
    np.random.seed(383628)
    mr = me.fit(n_rep=100)
    st = mr.summary()
    params_om = me.outcome_model.fit().params.to_numpy()
    p_om = [0.08118371, 0.96107436, 0.50801102, 1.22452252]
    assert_allclose(params_om, p_om, atol=1e-07)
    params_mm = me.mediator_model.fit().params.to_numpy()
    p_mm = [-0.0547506, 0.67478745, 17.03184275]
    assert_allclose(params_mm, p_mm, atol=1e-07)
    res_summ = np.array([[0.64539794, 0.57652012, 0.71427576, 0.0], [0.64539794, 0.57652012, 0.71427576, 0.0], [0.59401941, 0.56963807, 0.61840074, 0.0], [0.59401941, 0.56963807, 0.61840074, 0.0], [1.23941735, 1.1461582, 1.33267651, 0.0], [0.51935169, 0.50285723, 0.53584615, 0.0], [0.51935169, 0.50285723, 0.53584615, 0.0], [0.64539794, 0.57652012, 0.71427576, 0.0], [0.59401941, 0.56963807, 0.61840074, 0.0], [0.51935169, 0.50285723, 0.53584615, 0.0]])
    assert_allclose(st.to_numpy(), res_summ, rtol=0.15)
    assert_allclose(st.iloc[-1, 0], 0.56, rtol=0.01, atol=0.01)
    pm = st.loc['Prop. mediated (average)', 'Estimate']
    assert_allclose(pm, 0.56, rtol=0.01, atol=0.01)