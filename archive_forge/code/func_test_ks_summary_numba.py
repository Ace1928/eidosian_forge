import importlib
import numpy as np
import pytest
from ...data import load_arviz_data
from ...rcparams import rcParams
from ...stats import bfmi, mcse, rhat
from ...stats.diagnostics import _mc_error, ks_summary
from ...utils import Numba
from ..helpers import running_on_ci
from .test_diagnostics import data  # pylint: disable=unused-import
def test_ks_summary_numba():
    """Numba test for ks_summary."""
    state = Numba.numba_flag
    data = np.random.randn(100, 100)
    Numba.disable_numba()
    non_numba = ks_summary(data)['Count'].values
    Numba.enable_numba()
    with_numba = ks_summary(data)['Count'].values
    assert np.allclose(non_numba, with_numba)
    assert Numba.numba_flag == state