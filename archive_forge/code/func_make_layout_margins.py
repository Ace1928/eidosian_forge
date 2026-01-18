import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def make_layout_margins(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0, hspace=0, wspace=0):
    """
    For each axes, make a margin between the *pos* layoutbox and the
    *axes* layoutbox be a minimum size that can accommodate the
    decorations on the axis.

    Then make room for colorbars.

    Parameters
    ----------
    layoutgrids : dict
    fig : `~matplotlib.figure.Figure`
        `.Figure` instance to do the layout in.
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.
        The renderer to use.
    w_pad, h_pad : float, default: 0
        Width and height padding (in fraction of figure).
    hspace, wspace : float, default: 0
        Width and height padding as fraction of figure size divided by
        number of columns or rows.
    """
    for sfig in fig.subfigs:
        ss = sfig._subplotspec
        gs = ss.get_gridspec()
        make_layout_margins(layoutgrids, sfig, renderer, w_pad=w_pad, h_pad=h_pad, hspace=hspace, wspace=wspace)
        margins = get_margin_from_padding(sfig, w_pad=0, h_pad=0, hspace=hspace, wspace=wspace)
        layoutgrids[gs].edit_outer_margin_mins(margins, ss)
    for ax in fig._localaxes:
        if not ax.get_subplotspec() or not ax.get_in_layout():
            continue
        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()
        if gs not in layoutgrids:
            return
        margin = get_margin_from_padding(ax, w_pad=w_pad, h_pad=h_pad, hspace=hspace, wspace=wspace)
        pos, bbox = get_pos_and_bbox(ax, renderer)
        margin['left'] += pos.x0 - bbox.x0
        margin['right'] += bbox.x1 - pos.x1
        margin['bottom'] += pos.y0 - bbox.y0
        margin['top'] += bbox.y1 - pos.y1
        for cbax in ax._colorbars:
            pad = colorbar_get_pad(layoutgrids, cbax)
            cbp_rspan, cbp_cspan = get_cb_parent_spans(cbax)
            loc = cbax._colorbar_info['location']
            cbpos, cbbbox = get_pos_and_bbox(cbax, renderer)
            if loc == 'right':
                if cbp_cspan.stop == ss.colspan.stop:
                    margin['rightcb'] += cbbbox.width + pad
            elif loc == 'left':
                if cbp_cspan.start == ss.colspan.start:
                    margin['leftcb'] += cbbbox.width + pad
            elif loc == 'top':
                if cbp_rspan.start == ss.rowspan.start:
                    margin['topcb'] += cbbbox.height + pad
            elif cbp_rspan.stop == ss.rowspan.stop:
                margin['bottomcb'] += cbbbox.height + pad
            if loc in ['top', 'bottom']:
                if cbp_cspan.start == ss.colspan.start and cbbbox.x0 < bbox.x0:
                    margin['left'] += bbox.x0 - cbbbox.x0
                if cbp_cspan.stop == ss.colspan.stop and cbbbox.x1 > bbox.x1:
                    margin['right'] += cbbbox.x1 - bbox.x1
            if loc in ['left', 'right']:
                if cbp_rspan.stop == ss.rowspan.stop and cbbbox.y0 < bbox.y0:
                    margin['bottom'] += bbox.y0 - cbbbox.y0
                if cbp_rspan.start == ss.rowspan.start and cbbbox.y1 > bbox.y1:
                    margin['top'] += cbbbox.y1 - bbox.y1
        layoutgrids[gs].edit_outer_margin_mins(margin, ss)
    for leg in fig.legends:
        inv_trans_fig = None
        if leg._outside_loc and leg._bbox_to_anchor is None:
            if inv_trans_fig is None:
                inv_trans_fig = fig.transFigure.inverted().transform_bbox
            bbox = inv_trans_fig(leg.get_tightbbox(renderer))
            w = bbox.width + 2 * w_pad
            h = bbox.height + 2 * h_pad
            legendloc = leg._outside_loc
            if legendloc == 'lower':
                layoutgrids[fig].edit_margin_min('bottom', h)
            elif legendloc == 'upper':
                layoutgrids[fig].edit_margin_min('top', h)
            if legendloc == 'right':
                layoutgrids[fig].edit_margin_min('right', w)
            elif legendloc == 'left':
                layoutgrids[fig].edit_margin_min('left', w)