import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def reposition_axes(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0, hspace=0, wspace=0):
    """
    Reposition all the axes based on the new inner bounding box.
    """
    trans_fig_to_subfig = fig.transFigure - fig.transSubfigure
    for sfig in fig.subfigs:
        bbox = layoutgrids[sfig].get_outer_bbox()
        sfig._redo_transform_rel_fig(bbox=bbox.transformed(trans_fig_to_subfig))
        reposition_axes(layoutgrids, sfig, renderer, w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)
    for ax in fig._localaxes:
        if ax.get_subplotspec() is None or not ax.get_in_layout():
            continue
        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()
        if gs not in layoutgrids:
            return
        bbox = layoutgrids[gs].get_inner_bbox(rows=ss.rowspan, cols=ss.colspan)
        newbbox = trans_fig_to_subfig.transform_bbox(bbox)
        ax._set_position(newbbox)
        offset = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}
        for nn, cbax in enumerate(ax._colorbars[::-1]):
            if ax == cbax._colorbar_info['parents'][0]:
                reposition_colorbar(layoutgrids, cbax, renderer, offset=offset)