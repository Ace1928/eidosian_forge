from __future__ import annotations
from math import ceil
from typing import TYPE_CHECKING
import warnings
from matplotlib import ticker
import matplotlib.table
import numpy as np
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
def format_date_labels(ax: Axes, rot) -> None:
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
        label.set_rotation(rot)
    fig = ax.get_figure()
    if fig is not None:
        maybe_adjust_figure(fig, bottom=0.2)