import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
def logpdf(self, x):
    """
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function. The last axis of `x` must correspond
            to unit vectors of the same dimensionality as the distribution.

        Returns
        -------
        logpdf : ndarray or scalar
            Log of probability density function evaluated at `x`.

        """
    return self._dist._logpdf(x, self.dim, self.mu, self.kappa)