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
def sosfilt_zi(sos):
    """
    Construct initial conditions for sosfilt for step response steady-state.

    Compute an initial state `zi` for the `sosfilt` function that corresponds
    to the steady state of the step response.

    A typical use of this function is to set the initial state so that the
    output of the filter starts at the same value as the first element of
    the signal to be filtered.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    zi : ndarray
        Initial conditions suitable for use with ``sosfilt``, shape
        ``(n_sections, 2)``.

    See Also
    --------
    sosfilt, zpk2sos

    Notes
    -----
    .. versionadded:: 0.16.0

    Examples
    --------
    Filter a rectangular pulse that begins at time 0, with and without
    the use of the `zi` argument of `scipy.signal.sosfilt`.

    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> sos = signal.butter(9, 0.125, output='sos')
    >>> zi = signal.sosfilt_zi(sos)
    >>> x = (np.arange(250) < 100).astype(int)
    >>> f1 = signal.sosfilt(sos, x)
    >>> f2, zo = signal.sosfilt(sos, x, zi=zi)

    >>> plt.plot(x, 'k--', label='x')
    >>> plt.plot(f1, 'b', alpha=0.5, linewidth=2, label='filtered')
    >>> plt.plot(f2, 'g', alpha=0.25, linewidth=4, label='filtered with zi')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    sos = np.asarray(sos)
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError('sos must be shape (n_sections, 6)')
    if sos.dtype.kind in 'bui':
        sos = sos.astype(np.float64)
    n_sections = sos.shape[0]
    zi = np.empty((n_sections, 2), dtype=sos.dtype)
    scale = 1.0
    for section in range(n_sections):
        b = sos[section, :3]
        a = sos[section, 3:]
        zi[section] = scale * lfilter_zi(b, a)
        scale *= b.sum() / a.sum()
    return zi