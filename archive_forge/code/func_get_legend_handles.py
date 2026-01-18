from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.figure import Figure
from seaborn.utils import _version_predates
def get_legend_handles(legend):
    """Handle legendHandles attribute rename."""
    if _version_predates(mpl, '3.7'):
        return legend.legendHandles
    else:
        return legend.legend_handles