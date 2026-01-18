import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def check_no_collapsed_axes(layoutgrids, fig):
    """
    Check that no axes have collapsed to zero size.
    """
    for sfig in fig.subfigs:
        ok = check_no_collapsed_axes(layoutgrids, sfig)
        if not ok:
            return False
    for ax in fig.axes:
        gs = ax.get_gridspec()
        if gs in layoutgrids:
            lg = layoutgrids[gs]
            for i in range(gs.nrows):
                for j in range(gs.ncols):
                    bb = lg.get_inner_bbox(i, j)
                    if bb.width <= 0 or bb.height <= 0:
                        return False
    return True