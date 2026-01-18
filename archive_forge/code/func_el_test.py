import numpy as np
from scipy import optimize
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS, RegressionResults
from statsmodels.tools.tools import add_constant
def el_test(self, b0_vals, param_nums, method='nm', stochastic_exog=1, return_weights=0):
    """
        Returns the llr and p-value for a hypothesized parameter value
        for a regression that goes through the origin.

        Parameters
        ----------
        b0_vals : 1darray
            The hypothesized value to be tested.

        param_num : 1darray
            Which parameters to test.  Note this uses python
            indexing but the '0' parameter refers to the intercept term,
            which is assumed 0.  Therefore, param_num should be > 0.

        return_weights : bool
            If true, returns the weights that optimize the likelihood
            ratio at b0_vals.  Default is False.

        method : str
            Can either be 'nm' for Nelder-Mead or 'powell' for Powell.  The
            optimization method that optimizes over nuisance parameters.
            Default is 'nm'.

        stochastic_exog : bool
            When TRUE, the exogenous variables are assumed to be stochastic.
            When the regressors are nonstochastic, moment conditions are
            placed on the exogenous variables.  Confidence intervals for
            stochastic regressors are at least as large as non-stochastic
            regressors.  Default is TRUE.

        Returns
        -------
        res : tuple
            pvalue and likelihood ratio.
        """
    b0_vals = np.hstack((0, b0_vals))
    param_nums = np.hstack((0, param_nums))
    test_res = self.model.fit().el_test(b0_vals, param_nums, method=method, stochastic_exog=stochastic_exog, return_weights=return_weights)
    llr_test = test_res[0]
    llr_res = llr_test - self.llr
    pval = chi2.sf(llr_res, self.model.exog.shape[1] - 1)
    if return_weights:
        return (llr_res, pval, test_res[2])
    else:
        return (llr_res, pval)