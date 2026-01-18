import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation
from pandas import DataFrame
from scipy.stats import gaussian_kde, norm
import xarray as xr
from ...data import from_dict, load_arviz_data
from ...plots import (
from ...rcparams import rc_context, rcParams
from ...stats import compare, hdi, loo, waic
from ...stats.density_utils import kde as _kde
from ...utils import _cov
from ...plots.plot_utils import plot_point_interval
from ...plots.dotplot import wilkinson_algorithm
from ..helpers import (  # pylint: disable=unused-import
@pytest.mark.skipif(not animation.writers.is_available('ffmpeg'), reason='matplotlib animations within ArviZ require ffmpeg')
@pytest.mark.parametrize('system', ['Windows', 'Darwin'])
def test_non_linux_blit(models, monkeypatch, system, caplog):
    import platform

    def mock_system():
        return system
    monkeypatch.setattr(platform, 'system', mock_system)
    animation_kwargs = {'blit': True}
    axes, anim = plot_ppc(models.model_1, kind='kde', animated=True, animation_kwargs=animation_kwargs, num_pp_samples=5, random_seed=3)
    records = caplog.records
    assert len(records) == 1
    assert records[0].levelname == 'WARNING'
    assert axes
    assert anim