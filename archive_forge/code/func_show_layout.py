import re
import numpy as np
from pandas import DataFrame
from ...rcparams import rcParams
def show_layout(ax, show=True, force_layout=False):
    """Create a layout and call bokeh show."""
    if show is None:
        show = rcParams['plot.bokeh.show']
    if show:
        import bokeh.plotting as bkp
        layout = create_layout(ax, force_layout=force_layout)
        bkp.show(layout)