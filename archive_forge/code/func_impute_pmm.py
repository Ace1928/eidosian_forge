import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def impute_pmm(self, vname):
    """
        Use predictive mean matching to impute missing values.

        Notes
        -----
        The `perturb_params` method must be called first to define the
        model.
        """
    k_pmm = self.k_pmm
    endog_obs, exog_obs, exog_miss, predict_obs_kwds, predict_miss_kwds = self.get_split_data(vname)
    model = self.models[vname]
    pendog_obs = model.predict(self.params[vname], exog_obs, **predict_obs_kwds)
    pendog_miss = model.predict(self.params[vname], exog_miss, **predict_miss_kwds)
    pendog_obs = self._get_predicted(pendog_obs)
    pendog_miss = self._get_predicted(pendog_miss)
    ii = np.argsort(pendog_obs)
    endog_obs = endog_obs[ii]
    pendog_obs = pendog_obs[ii]
    ix = np.searchsorted(pendog_obs, pendog_miss)
    ixm = ix[:, None] + np.arange(-k_pmm, k_pmm)[None, :]
    msk = np.nonzero((ixm < 0) | (ixm > len(endog_obs) - 1))
    ixm = np.clip(ixm, 0, len(endog_obs) - 1)
    dx = pendog_miss[:, None] - pendog_obs[ixm]
    dx = np.abs(dx)
    dx[msk] = np.inf
    dxi = np.argsort(dx, 1)[:, 0:k_pmm]
    ir = np.random.randint(0, k_pmm, len(pendog_miss))
    jj = np.arange(dxi.shape[0])
    ix = dxi[jj, ir]
    iz = ixm[jj, ix]
    imputed_miss = np.array(endog_obs[iz]).squeeze()
    self._store_changes(vname, imputed_miss)