from __future__ import annotations
import os
from typing import Callable
import pandas._config.config as cf
from pandas._config.config import (
def register_plotting_backend_cb(key) -> None:
    if key == 'matplotlib':
        return
    from pandas.plotting._core import _get_plot_backend
    _get_plot_backend(key)