from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
def spm_hrf(RT, P=None, fMRI_T=16):
    """
    python implementation of spm_hrf

    See ``spm_hrf`` for implementation details::

      % RT   - scan repeat time
      % p    - parameters of the response function (two gamma
      % functions)
      % defaults  (seconds)
      % p(0) - delay of response (relative to onset)       6
      % p(1) - delay of undershoot (relative to onset)    16
      % p(2) - dispersion of response                      1
      % p(3) - dispersion of undershoot                    1
      % p(4) - ratio of response to undershoot             6
      % p(5) - onset (seconds)                             0
      % p(6) - length of kernel (seconds)                 32
      %
      % hrf  - hemodynamic response function
      % p    - parameters of the response function

    The following code using ``scipy.stats.distributions.gamma``
    doesn't return the same result as the ``spm_Gpdf`` function::

        hrf = gamma.pdf(u, p[0]/p[2], scale=dt/p[2]) -
              gamma.pdf(u, p[1]/p[3], scale=dt/p[3])/p[4]

    Example
    -------
    >>> print(spm_hrf(2))
    [  0.00000000e+00   8.65660810e-02   3.74888236e-01   3.84923382e-01
       2.16117316e-01   7.68695653e-02   1.62017720e-03  -3.06078117e-02
      -3.73060781e-02  -3.08373716e-02  -2.05161334e-02  -1.16441637e-02
      -5.82063147e-03  -2.61854250e-03  -1.07732374e-03  -4.10443522e-04
      -1.46257507e-04]

    """
    from scipy.special import gammaln
    p = np.array([6, 16, 1, 1, 6, 0, 32], dtype=float)
    if P is not None:
        p[0:len(P)] = P
    _spm_Gpdf = lambda x, h, l: np.exp(h * np.log(l) + (h - 1) * np.log(x) - l * x - gammaln(h))
    dt = RT / float(fMRI_T)
    u = np.arange(0, int(p[6] / dt + 1)) - p[5] / dt
    with np.errstate(divide='ignore'):
        hrf = _spm_Gpdf(u, p[0] / p[2], dt / p[2]) - _spm_Gpdf(u, p[1] / p[3], dt / p[3]) / p[4]
    idx = np.arange(0, int(p[6] / RT + 1)) * fMRI_T
    hrf = hrf[idx]
    hrf = hrf / np.sum(hrf)
    return hrf