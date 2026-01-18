import os
import nibabel as nb
import numpy as np
import pytest
from ...testing import utils
from ..confounds import CompCor, TCompCor, ACompCor
def close_up_to_column_sign(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """SVD can produce sign flips on a per-column basis."""
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    kwargs = dict(rtol=rtol, atol=atol, equal_nan=equal_nan)
    if np.allclose(a, b, **kwargs):
        return True
    ret = True
    for acol, bcol in zip(a.T, b.T):
        ret &= np.allclose(acol, bcol, **kwargs) or np.allclose(acol, -bcol, **kwargs)
        if not ret:
            break
    return ret