import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
def plot_cum_effects(self, orth=False, *, impulse=None, response=None, signif=0.05, plot_params=None, figsize=(10, 10), subplot_params=None, plot_stderr=True, stderr_type='asym', repl=1000, seed=None):
    """
        Plot cumulative impulse response functions

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse responses
        impulse : {str, int}
            variable providing the impulse
        response : {str, int}
            variable affected by the impulse
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        subplot_params : dict
            To pass to subplot plotting funcions. Example: if fonts are too big,
            pass {'fontsize' : 8} or some number to your taste.
        plot_params : dict

        figsize: (float, float), default (10, 10)
            Figure size (width, height in inches)
        plot_stderr : bool, default True
            Plot standard impulse response error bands
        stderr_type : str
            'asym': default, computes asymptotic standard errors
            'mc': monte carlo standard errors (use rpl)
        repl : int, default 1000
            Number of replications for monte carlo standard errors
        seed : int
            np.random.seed for Monte Carlo replications
        """
    if orth:
        title = 'Cumulative responses responses (orthogonalized)'
        cum_effects = self.orth_cum_effects
        lr_effects = self.orth_lr_effects
    else:
        title = 'Cumulative responses'
        cum_effects = self.cum_effects
        lr_effects = self.lr_effects
    if stderr_type not in ['asym', 'mc']:
        raise ValueError("`stderr_type` must be one of 'asym', 'mc'")
    else:
        if stderr_type == 'asym':
            stderr = self.cum_effect_cov(orth=orth)
        if stderr_type == 'mc':
            stderr = self.cum_errband_mc(orth=orth, repl=repl, signif=signif, seed=seed)
    if not plot_stderr:
        stderr = None
    fig = plotting.irf_grid_plot(cum_effects, stderr, impulse, response, self.model.names, title, signif=signif, hlines=lr_effects, subplot_params=subplot_params, plot_params=plot_params, figsize=figsize, stderr_type=stderr_type)
    return fig