import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.tools import pca
from statsmodels.sandbox.tools.cross_val import LeaveOneOut
def fit_find_nfact(self, maxfact=None, skip_crossval=True, cv_iter=None):
    """estimate the model and selection criteria for up to maxfact factors

        The selection criteria that are calculated are AIC, BIC, and R2_adj. and
        additionally cross-validation prediction error sum of squares if `skip_crossval`
        is false. Cross-validation is not used by default because it can be
        time consuming to calculate.

        By default the cross-validation method is Leave-one-out on the full dataset.
        A different cross-validation sample can be specified as an argument to
        cv_iter.

        Results are attached in `results_find_nfact`



        """
    if not hasattr(self, 'factors'):
        self.calc_factors()
    hasconst = self.hasconst
    if maxfact is None:
        maxfact = self.factors.shape[1] - hasconst
    if maxfact + hasconst < 1:
        raise ValueError('nothing to do, number of factors (incl. constant) should ' + 'be at least 1')
    maxfact = min(maxfact, 10)
    y0 = self.endog
    results = []
    for k in range(1, maxfact + hasconst):
        fact = self.factors[:, :k]
        res = sm.OLS(y0, fact).fit()
        if not skip_crossval:
            if cv_iter is None:
                cv_iter = LeaveOneOut(len(y0))
            prederr2 = 0.0
            for inidx, outidx in cv_iter:
                res_l1o = sm.OLS(y0[inidx], fact[inidx, :]).fit()
                prederr2 += (y0[outidx] - res_l1o.model.predict(res_l1o.params, fact[outidx, :])) ** 2.0
        else:
            prederr2 = np.nan
        results.append([k, res.aic, res.bic, res.rsquared_adj, prederr2])
    self.results_find_nfact = results = np.array(results)
    self.best_nfact = np.r_[np.argmin(results[:, 1:3], 0), np.argmax(results[:, 3], 0), np.argmin(results[:, -1], 0)]