import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
def ipw(self, return_results=True, effect_group='all', disp=False):
    """Inverse Probability Weighted treatment effect estimation.

        Parameters
        ----------
        return_results : bool
            If True, then a results instance is returned.
            If False, just ATE, POM0 and POM1 are returned.
        effect_group : {"all", 0, 1}
            ``effectgroup`` determines for which population the effects are
            estimated.
            If effect_group is "all", then sample average treatment effect and
            potential outcomes are returned.
            If effect_group is 1 or "treated", then effects on treated are
            returned.
            If effect_group is 0, "treated" or "control", then effects on
            untreated, i.e. control group, are returned.
        disp : bool
            Indicates whether the scipy optimizer should display the
            optimization results

        Returns
        -------
        TreatmentEffectsResults instance or tuple (ATE, POM0, POM1)

        See Also
        --------
        TreatmentEffectsResults
        """
    endog = self.model_pool.endog
    tind = self.treatment
    prob = self.prob_select
    if effect_group == 'all':
        probt = None
    elif effect_group in [1, 'treated']:
        probt = prob
        effect_group = 1
    elif effect_group in [0, 'untreated', 'control']:
        probt = 1 - prob
        effect_group = 0
    elif isinstance(effect_group, np.ndarray):
        probt = effect_group
        effect_group = 'user'
    else:
        raise ValueError('incorrect option for effect_group')
    res_ipw = ate_ipw(endog, tind, prob, weighted=True, probt=probt)
    if not return_results:
        return res_ipw
    gmm = _IPWGMM(endog, self.results_select, None, teff=self, effect_group=effect_group)
    start_params = np.concatenate((res_ipw[:2], self.results_select.params))
    res_gmm = gmm.fit(start_params=start_params, inv_weights=np.eye(len(start_params)), optim_method='nm', optim_args={'maxiter': 5000, 'disp': disp}, maxiter=1)
    res = TreatmentEffectResults(self, res_gmm, 'IPW', start_params=start_params, effect_group=effect_group)
    return res