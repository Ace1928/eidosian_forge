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
def display_rope(max_data):
    if rope is None:
        return
    elif isinstance(rope, dict):
        vals = None
        for sel in rope.get(var_name, []):
            if all((k in selection and selection[k] == v for k, v in sel.items() if k != 'rope')):
                vals = sel['rope']
                break
        if vals is None:
            return
    elif len(rope) == 2:
        vals = rope
    else:
        raise ValueError('Argument `rope` must be None, a dictionary like{"var_name": {"rope": (lo, hi)}}, or aniterable of length 2')
    rope_text = [f'{val:.{format_sig_figs(val, round_to)}g}' for val in vals]
    ax.line(vals, (max_data * 0.02, max_data * 0.02), line_width=linewidth * 5, line_color=vectorized_to_hex(rope_color), line_alpha=0.7)
    probability_within_rope = ((values > vals[0]) & (values <= vals[1])).mean()
    text_props = dict(text_color=vectorized_to_hex(rope_color), text_align='center')
    ax.text(x=values.mean(), y=[max_data * 0.45], text=[f'{format_as_percent(probability_within_rope, 1)} in ROPE'], **text_props)
    ax.text(x=vals, y=[max_data * 0.2, max_data * 0.2], text_font_size=f'{ax_labelsize}pt', text=rope_text, **text_props)