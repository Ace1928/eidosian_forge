import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from scipy import stats
from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import (
def jansen_sobol(f_A, f_B, f_AB):
    """Jansen for S and Sobol' for St.

            From Saltelli2010, table 2 formulations (c) and (e)."""
    var = np.var([f_A, f_B], axis=(0, -1))
    s = (var - 0.5 * np.mean((f_B - f_AB) ** 2, axis=-1)) / var
    st = np.mean(f_A * (f_A - f_AB), axis=-1) / var
    return (s.T, st.T)