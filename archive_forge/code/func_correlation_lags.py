import operator
import math
from math import prod as _prod
import timeit
import warnings
from scipy.spatial import cKDTree
from . import _sigtools
from ._ltisys import dlti
from ._upfirdn import upfirdn, _output_len, _upfirdn_modes
from scipy import linalg, fft as sp_fft
from scipy import ndimage
from scipy.fft._helper import _init_nd_shape_and_axes
import numpy as np
from scipy.special import lambertw
from .windows import get_window
from ._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext
from ._filter_design import cheby1, _validate_sos, zpk2sos
from ._fir_filter_design import firwin
from ._sosfilt import _sosfilt
def correlation_lags(in1_len, in2_len, mode='full'):
    """
    Calculates the lag / displacement indices array for 1D cross-correlation.

    Parameters
    ----------
    in1_len : int
        First input size.
    in2_len : int
        Second input size.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `correlate` for more information.

    Returns
    -------
    lags : array
        Returns an array containing cross-correlation lag/displacement indices.
        Indices can be indexed with the np.argmax of the correlation to return
        the lag/displacement.

    See Also
    --------
    correlate : Compute the N-dimensional cross-correlation.

    Notes
    -----
    Cross-correlation for continuous functions :math:`f` and :math:`g` is
    defined as:

    .. math::

        \\left ( f\\star g \\right )\\left ( \\tau \\right )
        \\triangleq \\int_{t_0}^{t_0 +T}
        \\overline{f\\left ( t \\right )}g\\left ( t+\\tau \\right )dt

    Where :math:`\\tau` is defined as the displacement, also known as the lag.

    Cross correlation for discrete functions :math:`f` and :math:`g` is
    defined as:

    .. math::
        \\left ( f\\star g \\right )\\left [ n \\right ]
        \\triangleq \\sum_{-\\infty}^{\\infty}
        \\overline{f\\left [ m \\right ]}g\\left [ m+n \\right ]

    Where :math:`n` is the lag.

    Examples
    --------
    Cross-correlation of a signal with its time-delayed self.

    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal(1000)
    >>> y = np.concatenate([rng.standard_normal(100), x])
    >>> correlation = signal.correlate(x, y, mode="full")
    >>> lags = signal.correlation_lags(x.size, y.size, mode="full")
    >>> lag = lags[np.argmax(correlation)]
    """
    if mode == 'full':
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == 'same':
        lags = np.arange(-in2_len + 1, in1_len)
        mid = lags.size // 2
        lag_bound = in1_len // 2
        if in1_len % 2 == 0:
            lags = lags[mid - lag_bound:mid + lag_bound]
        else:
            lags = lags[mid - lag_bound:mid + lag_bound + 1]
    elif mode == 'valid':
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags