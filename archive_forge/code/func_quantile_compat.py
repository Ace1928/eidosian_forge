from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.core.dtypes.missing import (
def quantile_compat(values: ArrayLike, qs: npt.NDArray[np.float64], interpolation: str) -> ArrayLike:
    """
    Compute the quantiles of the given values for each quantile in `qs`.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    qs : np.ndarray[float64]
    interpolation : str

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    if isinstance(values, np.ndarray):
        fill_value = na_value_for_dtype(values.dtype, compat=False)
        mask = isna(values)
        return quantile_with_mask(values, mask, fill_value, qs, interpolation)
    else:
        return values._quantile(qs, interpolation)