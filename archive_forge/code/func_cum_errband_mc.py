import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
def cum_errband_mc(self, orth=False, repl=1000, signif=0.05, seed=None, burn=100):
    """
        IRF Monte Carlo integrated error bands of cumulative effect
        """
    model = self.model
    periods = self.periods
    return model.irf_errband_mc(orth=orth, repl=repl, steps=periods, signif=signif, seed=seed, burn=burn, cum=True)