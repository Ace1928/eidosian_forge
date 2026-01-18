import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def profile_re(self, re_ix, vtype, num_low=5, dist_low=1.0, num_high=5, dist_high=1.0, **fit_kwargs):
    """
        Profile-likelihood inference for variance parameters.

        Parameters
        ----------
        re_ix : int
            If vtype is `re`, this value is the index of the variance
            parameter for which to construct a profile likelihood.  If
            `vtype` is 'vc' then `re_ix` is the name of the variance
            parameter to be profiled.
        vtype : str
            Either 're' or 'vc', depending on whether the profile
            analysis is for a random effect or a variance component.
        num_low : int
            The number of points at which to calculate the likelihood
            below the MLE of the parameter of interest.
        dist_low : float
            The distance below the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        num_high : int
            The number of points at which to calculate the likelihood
            above the MLE of the parameter of interest.
        dist_high : float
            The distance above the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        **fit_kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        An array with two columns.  The first column contains the
        values to which the parameter of interest is constrained.  The
        second column contains the corresponding likelihood values.

        Notes
        -----
        Only variance parameters can be profiled.
        """
    pmodel = self.model
    k_fe = pmodel.k_fe
    k_re = pmodel.k_re
    k_vc = pmodel.k_vc
    endog, exog = (pmodel.endog, pmodel.exog)
    if vtype == 're':
        ix = np.arange(k_re)
        ix[0] = re_ix
        ix[re_ix] = 0
        exog_re = pmodel.exog_re.copy()[:, ix]
        params = self.params_object.copy()
        cov_re_unscaled = params.cov_re
        cov_re_unscaled = cov_re_unscaled[np.ix_(ix, ix)]
        params.cov_re = cov_re_unscaled
        ru0 = cov_re_unscaled[0, 0]
        cov_re = self.scale * cov_re_unscaled
        low = (cov_re[0, 0] - dist_low) / self.scale
        high = (cov_re[0, 0] + dist_high) / self.scale
    elif vtype == 'vc':
        re_ix = self.model.exog_vc.names.index(re_ix)
        params = self.params_object.copy()
        vcomp = self.vcomp
        low = (vcomp[re_ix] - dist_low) / self.scale
        high = (vcomp[re_ix] + dist_high) / self.scale
        ru0 = vcomp[re_ix] / self.scale
    if low <= 0:
        raise ValueError('dist_low is too large and would result in a negative variance. Try a smaller value.')
    left = np.linspace(low, ru0, num_low + 1)
    right = np.linspace(ru0, high, num_high + 1)[1:]
    rvalues = np.concatenate((left, right))
    free = MixedLMParams(k_fe, k_re, k_vc)
    if self.freepat is None:
        free.fe_params = np.ones(k_fe)
        vcomp = np.ones(k_vc)
        mat = np.ones((k_re, k_re))
    else:
        free.fe_params = self.freepat.fe_params
        vcomp = self.freepat.vcomp
        mat = self.freepat.cov_re
        if vtype == 're':
            mat = mat[np.ix_(ix, ix)]
    if vtype == 're':
        mat[0, 0] = 0
    else:
        vcomp[re_ix] = 0
    free.cov_re = mat
    free.vcomp = vcomp
    klass = self.model.__class__
    init_kwargs = pmodel._get_init_kwds()
    if vtype == 're':
        init_kwargs['exog_re'] = exog_re
    likev = []
    for x in rvalues:
        model = klass(endog, exog, **init_kwargs)
        if vtype == 're':
            cov_re = params.cov_re.copy()
            cov_re[0, 0] = x
            params.cov_re = cov_re
        else:
            params.vcomp[re_ix] = x
        rslt = model.fit(start_params=params, free=free, reml=self.reml, cov_pen=self.cov_pen, **fit_kwargs)._results
        likev.append([x * rslt.scale, rslt.llf])
    likev = np.asarray(likev)
    return likev