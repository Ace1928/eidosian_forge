import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def get_cb_parent_spans(cbax):
    """
    Figure out which subplotspecs this colorbar belongs to.

    Parameters
    ----------
    cbax : `~matplotlib.axes.Axes`
        Axes for the colorbar.
    """
    rowstart = np.inf
    rowstop = -np.inf
    colstart = np.inf
    colstop = -np.inf
    for parent in cbax._colorbar_info['parents']:
        ss = parent.get_subplotspec()
        rowstart = min(ss.rowspan.start, rowstart)
        rowstop = max(ss.rowspan.stop, rowstop)
        colstart = min(ss.colspan.start, colstart)
        colstop = max(ss.colspan.stop, colstop)
    rowspan = range(rowstart, rowstop)
    colspan = range(colstart, colstop)
    return (rowspan, colspan)