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
def display_hdi(max_data):
    hdi_probs = hdi(values, hdi_prob=hdi_prob, circular=circular, multimodal=multimodal, skipna=skipna)
    for hdi_i in np.atleast_2d(hdi_probs):
        ax.line(hdi_i, (max_data * 0.02, max_data * 0.02), line_width=linewidth * 2, line_color='black')
        ax.text(x=list(hdi_i) + [(hdi_i[0] + hdi_i[1]) / 2], y=[max_data * 0.07, max_data * 0.07, max_data * 0.3], text=list(map(str, map(lambda x: round_num(x, round_to), hdi_i))) + [f'{format_as_percent(hdi_prob)} HDI'], text_align='center')