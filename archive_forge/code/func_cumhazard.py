import numpy as np
from scipy import integrate, stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.validation import array_like, float_like
from . import bandwidths
from .kdetools import forrt, revrt, silverman_transform
from .linbin import fast_linbin
@cache_readonly
def cumhazard(self):
    """
        Returns the hazard function evaluated at the support.

        Notes
        -----
        Will not work if fit has not been called.
        """
    _checkisfit(self)
    return -np.log(self.sf)