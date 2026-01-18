from statsmodels.sandbox.predict_functional import predict_functional
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
@pytest.mark.matplotlib
def test_glm_formula(self, close_figures):
    np.random.seed(542)
    n = 500
    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)
    x3 = np.random.randint(0, 3, size=n)
    x3 = np.asarray(['ABC'[i] for i in x3])
    lin_pred = -1 + 0.5 * x1 ** 2 + (x3 == 'B')
    prob = 1 / (1 + np.exp(-lin_pred))
    y = 1 * (np.random.uniform(size=n) < prob)
    df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3})
    fml = 'y ~ x1 + I(x1**2) + x2 + C(x3)'
    model = sm.GLM.from_formula(fml, family=sm.families.Binomial(), data=df)
    result = model.fit()
    summaries = {'x2': np.mean}
    for linear in (False, True):
        values = {'x3': 'B'}
        pr1, ci1, fvals1 = predict_functional(result, 'x1', summaries, values, linear=linear)
        values = {'x3': 'C'}
        pr2, ci2, fvals2 = predict_functional(result, 'x1', summaries, values, linear=linear)
        exact1 = -1 + 0.5 * fvals1 ** 2 + 1
        exact2 = -1 + 0.5 * fvals2 ** 2
        if not linear:
            exact1 = 1 / (1 + np.exp(-exact1))
            exact2 = 1 / (1 + np.exp(-exact2))
        plt.clf()
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.7, 0.8])
        plt.plot(fvals1, pr1, '-', label='x3=B')
        plt.plot(fvals2, pr2, '-', label='x3=C')
        plt.plot(fvals1, exact1, '-', label='x3=B (exact)')
        plt.plot(fvals2, exact2, '-', label='x3=C (exact)')
        ha, lb = ax.get_legend_handles_labels()
        plt.figlegend(ha, lb, loc='center right')
        plt.xlabel('Focus variable', size=15)
        if linear:
            plt.ylabel('Fitted linear predictor', size=15)
        else:
            plt.ylabel('Fitted probability', size=15)
        plt.title('Binomial GLM prediction')
        self.close_or_save(fig)
        plt.clf()
        fig = plt.figure()
        ax = plt.axes([0.1, 0.1, 0.7, 0.8])
        plt.plot(fvals1, pr1, '-', label='x3=B', color='orange')
        plt.fill_between(fvals1, ci1[:, 0], ci1[:, 1], color='grey')
        plt.plot(fvals2, pr2, '-', label='x3=C', color='lime')
        plt.fill_between(fvals2, ci2[:, 0], ci2[:, 1], color='grey')
        ha, lb = ax.get_legend_handles_labels()
        plt.figlegend(ha, lb, loc='center right')
        plt.xlabel('Focus variable', size=15)
        if linear:
            plt.ylabel('Fitted linear predictor', size=15)
        else:
            plt.ylabel('Fitted probability', size=15)
        plt.title('Binomial GLM prediction')
        self.close_or_save(fig)