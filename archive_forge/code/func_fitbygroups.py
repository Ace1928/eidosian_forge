import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS
def fitbygroups(self):
    """Fit OLS regression for each group separately.

        Returns
        -------
        results are attached

        olsbygroup : dictionary of result instance
            the returned regression results for each group
        sigmabygroup : array (ngroups,) (this should be called sigma2group ??? check)
            mse_resid for each group
        weights : array (nobs,)
            standard deviation of group extended to the original observations. This can
            be used as weights in WLS for group-wise heteroscedasticity.



        """
    olsbygroup = {}
    sigmabygroup = []
    for gi, group in enumerate(self.unique):
        groupmask = self.groupsint == gi
        res = OLS(self.endog[groupmask], self.exog[groupmask]).fit()
        olsbygroup[group] = res
        sigmabygroup.append(res.mse_resid)
    self.olsbygroup = olsbygroup
    self.sigmabygroup = np.array(sigmabygroup)
    self.weights = np.sqrt(self.sigmabygroup[self.groupsint])