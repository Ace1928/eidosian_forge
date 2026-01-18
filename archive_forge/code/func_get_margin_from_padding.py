import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def get_margin_from_padding(obj, *, w_pad=0, h_pad=0, hspace=0, wspace=0):
    ss = obj._subplotspec
    gs = ss.get_gridspec()
    if hasattr(gs, 'hspace'):
        _hspace = gs.hspace if gs.hspace is not None else hspace
        _wspace = gs.wspace if gs.wspace is not None else wspace
    else:
        _hspace = gs._hspace if gs._hspace is not None else hspace
        _wspace = gs._wspace if gs._wspace is not None else wspace
    _wspace = _wspace / 2
    _hspace = _hspace / 2
    nrows, ncols = gs.get_geometry()
    margin = {'leftcb': w_pad, 'rightcb': w_pad, 'bottomcb': h_pad, 'topcb': h_pad, 'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
    if _wspace / ncols > w_pad:
        if ss.colspan.start > 0:
            margin['leftcb'] = _wspace / ncols
        if ss.colspan.stop < ncols:
            margin['rightcb'] = _wspace / ncols
    if _hspace / nrows > h_pad:
        if ss.rowspan.stop < nrows:
            margin['bottomcb'] = _hspace / nrows
        if ss.rowspan.start > 0:
            margin['topcb'] = _hspace / nrows
    return margin