import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
def err_band_sz2(self, orth=False, svar=False, repl=1000, signif=0.05, seed=None, burn=100, component=None):
    """
        IRF Sims-Zha error band method 2.

        This method Does not assume symmetric error bands around mean.

        Parameters
        ----------
        orth : bool, default False
            Compute orthogonalized impulse responses
        repl : int, default 1000
            Number of MC replications
        signif : float (0 < signif < 1)
            Significance level for error bars, defaults to 95% CI
        seed : int, default None
            np.random seed
        burn : int, default 100
            Number of initial simulated obs to discard
        component : neqs x neqs array, default to largest for each
            Index of column of eigenvector/value to use for each error band
            Note: period of impulse (t=0) is not included when computing
            principle component

        References
        ----------
        Sims, Christopher A., and Tao Zha. 1999. "Error Bands for Impulse
        Response". Econometrica 67: 1113-1155.
        """
    model = self.model
    periods = self.periods
    irfs = self._choose_irfs(orth, svar)
    neqs = self.neqs
    irf_resim = model.irf_resim(orth=orth, repl=repl, steps=periods, seed=seed, burn=100)
    W, eigva, k = self._eigval_decomp_SZ(irf_resim)
    if component is not None:
        if np.shape(component) != (neqs, neqs):
            raise ValueError('Component array must be ' + str(neqs) + ' x ' + str(neqs))
        if np.argmax(component) >= neqs * periods:
            raise ValueError('Atleast one of the components does not exist')
        else:
            k = component
    gamma = np.zeros((repl, periods + 1, neqs, neqs))
    for p in range(repl):
        for i in range(neqs):
            for j in range(neqs):
                gamma[p, 1:, i, j] = W[i, j, k[i, j], :] * irf_resim[p, 1:, i, j]
    gamma_sort = np.sort(gamma, axis=0)
    indx = (round(signif / 2 * repl) - 1, round((1 - signif / 2) * repl) - 1)
    lower = np.copy(irfs)
    upper = np.copy(irfs)
    for i in range(neqs):
        for j in range(neqs):
            lower[:, i, j] = irfs[:, i, j] + gamma_sort[indx[0], :, i, j]
            upper[:, i, j] = irfs[:, i, j] + gamma_sort[indx[1], :, i, j]
    return (lower, upper)