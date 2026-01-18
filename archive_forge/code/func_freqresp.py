import warnings
import copy
from math import sqrt
import cupy
from cupyx.scipy import linalg
from cupyx.scipy.interpolate import make_interp_spline
from cupyx.scipy.linalg import expm, block_diag
from cupyx.scipy.signal._lti_conversion import (
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._filter_design import (
def freqresp(self, w=None, n=10000, whole=False):
    """
        Calculate the frequency response of a discrete-time system.

        Returns a 2-tuple containing arrays of frequencies [rad/s] and
        complex magnitude.
        See `dfreqresp` for details.

        """
    return dfreqresp(self, w=w, n=n, whole=whole)