from __future__ import annotations
import random
from typing import TYPE_CHECKING
from matplotlib import patches
import matplotlib.lines as mlines
import numpy as np
from pandas.core.dtypes.missing import notna
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
def lag_plot(series: Series, lag: int=1, ax: Axes | None=None, **kwds) -> Axes:
    import matplotlib.pyplot as plt
    kwds.setdefault('c', plt.rcParams['patch.facecolor'])
    data = series.values
    y1 = data[:-lag]
    y2 = data[lag:]
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel('y(t)')
    ax.set_ylabel(f'y(t + {lag})')
    ax.scatter(y1, y2, **kwds)
    return ax