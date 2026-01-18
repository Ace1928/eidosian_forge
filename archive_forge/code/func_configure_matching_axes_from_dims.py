import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def configure_matching_axes_from_dims(fig, matching_prop='_dim'):
    """
    Configure matching axes for a figure

    Note: This function mutates the input figure

    Parameters
    ----------
    fig: dict
        The figure dictionary to process.
    matching_prop: str
        The name of the axis property that should be used to determine that two axes
        should be matched together.  If the property is missing or None, axes will not
        be matched
    """
    axis_map = {}
    for k, v in fig.get('layout', {}).items():
        if k[1:5] == 'axis':
            matching_val = v.get(matching_prop, None)
            axis_map.setdefault(matching_val, [])
            axis_ref = k.replace('axis', '')
            axis_pair = (axis_ref, v)
            axis_map[matching_val].append(axis_pair)
    for _, axis_pairs in axis_map.items():
        if len(axis_pairs) < 2:
            continue
        matches_reference, linked_axis = axis_pairs[0]
        for _, axis in axis_pairs[1:]:
            axis['matches'] = matches_reference
            if 'range' in axis and 'range' in linked_axis:
                linked_axis['range'] = [v if isfinite(v) else None for v in max_range([axis['range'], linked_axis['range']])]