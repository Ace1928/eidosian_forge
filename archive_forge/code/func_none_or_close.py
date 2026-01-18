import operator
import warnings
import numpy as np
from nibabel.optpkg import optional_package
from ..onetime import auto_attr as one_time
from ..openers import ImageOpener
from . import csareader as csar
from .dwiparams import B2q, nearest_pos_semi_def, q2bg
def none_or_close(val1, val2, rtol=1e-05, atol=1e-06):
    """Match if `val1` and `val2` are both None, or are close

    Parameters
    ----------
    val1 : None or array-like
    val2 : None or array-like
    rtol : float, optional
       Relative tolerance; see ``np.allclose``
    atol : float, optional
       Absolute tolerance; see ``np.allclose``

    Returns
    -------
    tf : bool
       True iff (both `val1` and `val2` are None) or (`val1` and `val2`
       are close arrays, as detected by ``np.allclose`` with parameters
       `rtol` and `atal`).

    Examples
    --------
    >>> none_or_close(None, None)
    True
    >>> none_or_close(1, None)
    False
    >>> none_or_close(None, 1)
    False
    >>> none_or_close([1,2], [1,2])
    True
    >>> none_or_close([0,1], [0,2])
    False
    """
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False
    return np.allclose(val1, val2, rtol, atol)