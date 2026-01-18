import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type, check_nD
def filter_inverse(data, impulse_response=None, filter_params=None, max_gain=2, predefined_filter=None):
    """Apply the filter in reverse to the given data.

    Parameters
    ----------
    data : (M, N) ndarray
        Input data.
    impulse_response : callable `f(r, c, **filter_params)`
        Impulse response of the filter.  See :class:`~.LPIFilter2D`. This is a required
        argument unless a `predifined_filter` is provided.
    filter_params : dict, optional
        Additional keyword parameters to the impulse_response function.
    max_gain : float, optional
        Limit the filter gain.  Often, the filter contains zeros, which would
        cause the inverse filter to have infinite gain.  High gain causes
        amplification of artefacts, so a conservative limit is recommended.

    Other Parameters
    ----------------
    predefined_filter : LPIFilter2D, optional
        If you need to apply the same filter multiple times over different
        images, construct the LPIFilter2D and specify it here.

    """
    if filter_params is None:
        filter_params = {}
    check_nD(data, 2, 'data')
    if predefined_filter is None:
        filt = LPIFilter2D(impulse_response, **filter_params)
    else:
        filt = predefined_filter
    F, G = filt._prepare(data)
    _min_limit(F, val=np.finfo(F.real.dtype).eps)
    F = 1 / F
    mask = np.abs(F) > max_gain
    F[mask] = np.sign(F[mask]) * max_gain
    return _center(np.abs(fft.ifftshift(fft.ifftn(G * F))), data.shape)