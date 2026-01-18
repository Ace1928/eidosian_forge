import os
import inspect
import warnings
import colorsys
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve
from types import ModuleType
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from seaborn._core.typing import deprecated
from seaborn.external.version import Version
from seaborn.external.appdirs import user_cache_dir
def locator_to_legend_entries(locator, limits, dtype):
    """Return levels and formatted levels for brief numeric legends."""
    raw_levels = locator.tick_values(*limits).astype(dtype)
    raw_levels = [l for l in raw_levels if l >= limits[0] and l <= limits[1]]

    class dummy_axis:

        def get_view_interval(self):
            return limits
    if isinstance(locator, mpl.ticker.LogLocator):
        formatter = mpl.ticker.LogFormatter()
    else:
        formatter = mpl.ticker.ScalarFormatter()
        formatter.set_useOffset(False)
        formatter.set_scientific(False)
    formatter.axis = dummy_axis()
    formatted_levels = formatter.format_ticks(raw_levels)
    return (raw_levels, formatted_levels)