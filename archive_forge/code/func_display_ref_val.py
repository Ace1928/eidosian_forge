from numbers import Number
from typing import Optional
import numpy as np
from bokeh.models.annotations import Title
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram
from ...kdeplot import plot_kde
from ...plot_utils import (
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
def display_ref_val(max_data):
    if ref_val is None:
        return
    elif isinstance(ref_val, dict):
        val = None
        for sel in ref_val.get(var_name, []):
            if all((k in selection and selection[k] == v for k, v in sel.items() if k != 'ref_val')):
                val = sel['ref_val']
                break
        if val is None:
            return
    elif isinstance(ref_val, list):
        val = ref_val[idx]
    elif isinstance(ref_val, Number):
        val = ref_val
    else:
        raise ValueError('Argument `ref_val` must be None, a constant, a list or a dictionary like {"var_name": [{"ref_val": ref_val}]}')
    less_than_ref_probability = (values < val).mean()
    greater_than_ref_probability = (values >= val).mean()
    ref_in_posterior = '{} <{:g}< {}'.format(format_as_percent(less_than_ref_probability, 1), val, format_as_percent(greater_than_ref_probability, 1))
    ax.line([val, val], [0, 0.8 * max_data], line_color=vectorized_to_hex(ref_val_color), line_alpha=0.65)
    ax.text(x=[values.mean()], y=[max_data * 0.6], text=[ref_in_posterior], text_color=vectorized_to_hex(ref_val_color), text_align='center')