from statsmodels.sandbox.predict_functional import predict_functional
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
@pytest.mark.matplotlib
def test_noformula_prediction(self, close_figures):
    np.random.seed(6434)
    n = 200
    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)
    x3 = np.random.normal(size=n)
    y = x1 - x2 + np.random.normal(size=n)
    exog = np.vstack((x1, x2, x3)).T
    model = sm.OLS(y, exog)
    result = model.fit()
    summaries = {'x3': pctl(0.75)}
    values = {'x2': 1}
    pr1, ci1, fvals1 = predict_functional(result, 'x1', summaries, values)
    values = {'x2': -1}
    pr2, ci2, fvals2 = predict_functional(result, 'x1', summaries, values)
    plt.clf()
    fig = plt.figure()
    ax = plt.axes([0.1, 0.1, 0.7, 0.8])
    plt.plot(fvals1, pr1, '-', label='x2=1', lw=4, alpha=0.6, color='orange')
    plt.plot(fvals2, pr2, '-', label='x2=-1', lw=4, alpha=0.6, color='lime')
    ha, lb = ax.get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, loc='center right')
    leg.draw_frame(False)
    plt.xlabel('Focus variable', size=15)
    plt.ylabel('Fitted mean', size=15)
    plt.title('Linear model prediction')
    self.close_or_save(fig)
    plt.clf()
    fig = plt.figure()
    ax = plt.axes([0.1, 0.1, 0.7, 0.8])
    plt.plot(fvals1, pr1, '-', label='x2=1', lw=4, alpha=0.6, color='orange')
    plt.fill_between(fvals1, ci1[:, 0], ci1[:, 1], color='grey')
    plt.plot(fvals1, pr2, '-', label='x2=1', lw=4, alpha=0.6, color='lime')
    plt.fill_between(fvals2, ci2[:, 0], ci2[:, 1], color='grey')
    ha, lb = ax.get_legend_handles_labels()
    plt.figlegend(ha, lb, loc='center right')
    plt.xlabel('Focus variable', size=15)
    plt.ylabel('Fitted mean', size=15)
    plt.title('Linear model prediction')
    self.close_or_save(fig)