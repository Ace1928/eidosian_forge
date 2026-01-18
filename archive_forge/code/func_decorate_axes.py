from __future__ import annotations
import functools
from typing import (
import warnings
import numpy as np
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.converter import (
from pandas.tseries.frequencies import (
def decorate_axes(ax: Axes, freq: BaseOffset) -> None:
    """Initialize axes for time-series plotting"""
    if not hasattr(ax, '_plot_data'):
        ax._plot_data = []
    ax.freq = freq
    xaxis = ax.get_xaxis()
    xaxis.freq = freq