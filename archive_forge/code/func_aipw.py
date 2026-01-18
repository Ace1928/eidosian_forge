import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
@Substitution(params_returns=indent(doc_params_returns2, ' ' * 8))
def aipw(self, return_results=True, disp=False):
    """
        ATE and POM from double robust augmented inverse probability weighting
        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults

        """
    nobs = self.nobs
    prob = self.prob_select
    tind = self.treatment
    exog = self.model_pool.exog
    correct0 = (self.results0.resid / (1 - prob[tind == 0])).sum() / nobs
    correct1 = (self.results1.resid / prob[tind == 1]).sum() / nobs
    tmean0 = self.results0.predict(exog).mean() + correct0
    tmean1 = self.results1.predict(exog).mean() + correct1
    ate = tmean1 - tmean0
    if not return_results:
        return (ate, tmean0, tmean1)
    endog = self.model_pool.endog
    p2_aipw = np.asarray([ate, tmean0])
    mag_aipw1 = _AIPWGMM(endog, self.results_select, None, teff=self)
    start_params = np.concatenate((p2_aipw, self.results0.params, self.results1.params, self.results_select.params))
    res_gmm = mag_aipw1.fit(start_params=start_params, inv_weights=np.eye(len(start_params)), optim_method='nm', optim_args={'maxiter': 5000, 'disp': disp}, maxiter=1)
    res = TreatmentEffectResults(self, res_gmm, 'IPW', start_params=start_params, effect_group='all')
    return res