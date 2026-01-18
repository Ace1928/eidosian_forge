import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def sos2tf(sos):
    """
    Return a single transfer function from a series of second-order sections

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    See Also
    --------
    scipy.signal.sos2tf

    """
    sos = cupy.asarray(sos)
    result_type = sos.dtype
    if result_type.kind in 'bui':
        result_type = cupy.float64
    b = cupy.array([1], dtype=result_type)
    a = cupy.array([1], dtype=result_type)
    n_sections = sos.shape[0]
    for section in range(n_sections):
        b = cupy.polymul(b, sos[section, :3])
        a = cupy.polymul(a, sos[section, 3:])
    return (b, a)