import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def clean_internal_figure_properties(fig):
    """
    Remove all HoloViews internal properties (those with leading underscores) from the
    input figure.

    Note: This function mutates the input figure

    Parameters
    ----------
    fig: dict
        The figure dictionary to process.
    """
    fig_props = list(fig)
    for prop in fig_props:
        val = fig[prop]
        if prop.startswith('_'):
            fig.pop(prop)
        elif isinstance(val, dict):
            clean_internal_figure_properties(val)
        elif isinstance(val, (list, tuple)) and val and isinstance(val[0], dict):
            for el in val:
                clean_internal_figure_properties(el)