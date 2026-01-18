import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def make_layoutgrids(fig, layoutgrids, rect=(0, 0, 1, 1)):
    """
    Make the layoutgrid tree.

    (Sub)Figures get a layoutgrid so we can have figure margins.

    Gridspecs that are attached to axes get a layoutgrid so axes
    can have margins.
    """
    if layoutgrids is None:
        layoutgrids = dict()
        layoutgrids['hasgrids'] = False
    if not hasattr(fig, '_parent'):
        layoutgrids[fig] = mlayoutgrid.LayoutGrid(parent=rect, name='figlb')
    else:
        gs = fig._subplotspec.get_gridspec()
        layoutgrids = make_layoutgrids_gs(layoutgrids, gs)
        parentlb = layoutgrids[gs]
        layoutgrids[fig] = mlayoutgrid.LayoutGrid(parent=parentlb, name='panellb', parent_inner=True, nrows=1, ncols=1, parent_pos=(fig._subplotspec.rowspan, fig._subplotspec.colspan))
    for sfig in fig.subfigs:
        layoutgrids = make_layoutgrids(sfig, layoutgrids)
    for ax in fig._localaxes:
        gs = ax.get_gridspec()
        if gs is not None:
            layoutgrids = make_layoutgrids_gs(layoutgrids, gs)
    return layoutgrids