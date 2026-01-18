import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from scipy import stats
from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import (
def jansen_sobol_typed(f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return jansen_sobol(f_A, f_B, f_AB)