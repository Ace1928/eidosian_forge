import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def reset_margins(layoutgrids, fig):
    """
    Reset the margins in the layoutboxes of *fig*.

    Margins are usually set as a minimum, so if the figure gets smaller
    the minimum needs to be zero in order for it to grow again.
    """
    for sfig in fig.subfigs:
        reset_margins(layoutgrids, sfig)
    for ax in fig.axes:
        if ax.get_in_layout():
            gs = ax.get_gridspec()
            if gs in layoutgrids:
                layoutgrids[gs].reset_margins()
    layoutgrids[fig].reset_margins()