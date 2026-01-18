from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import (
from .palettes import color_palette, blend_palette
from ._docstrings import (
def set_yticklabels(self, labels=None, **kwargs):
    """Set y axis tick labels on the left column of the grid."""
    for ax in self.axes.flat:
        curr_ticks = ax.get_yticks()
        ax.set_yticks(curr_ticks)
        if labels is None:
            curr_labels = [label.get_text() for label in ax.get_yticklabels()]
            ax.set_yticklabels(curr_labels, **kwargs)
        else:
            ax.set_yticklabels(labels, **kwargs)
    return self