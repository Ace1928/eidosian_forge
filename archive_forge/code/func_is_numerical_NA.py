import numpy as np
from patsy import PatsyError
from patsy.util import (safe_isnan, safe_scalar_isnan,
def is_numerical_NA(self, arr):
    """Returns a 1-d mask array indicating which rows in an array of
        numerical values contain at least one NA value.

        Note that here `arr` is a numpy array or pandas DataFrame."""
    mask = np.zeros(arr.shape, dtype=bool)
    if 'NaN' in self.NA_types:
        mask |= np.isnan(arr)
    if mask.ndim > 1:
        mask = np.any(mask, axis=1)
    return mask