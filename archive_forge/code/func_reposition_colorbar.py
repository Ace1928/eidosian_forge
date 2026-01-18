import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def reposition_colorbar(layoutgrids, cbax, renderer, *, offset=None):
    """
    Place the colorbar in its new place.

    Parameters
    ----------
    layoutgrids : dict
    cbax : `~matplotlib.axes.Axes`
        Axes for the colorbar.
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.
        The renderer to use.
    offset : array-like
        Offset the colorbar needs to be pushed to in order to
        account for multiple colorbars.
    """
    parents = cbax._colorbar_info['parents']
    gs = parents[0].get_gridspec()
    fig = cbax.figure
    trans_fig_to_subfig = fig.transFigure - fig.transSubfigure
    cb_rspans, cb_cspans = get_cb_parent_spans(cbax)
    bboxparent = layoutgrids[gs].get_bbox_for_cb(rows=cb_rspans, cols=cb_cspans)
    pb = layoutgrids[gs].get_inner_bbox(rows=cb_rspans, cols=cb_cspans)
    location = cbax._colorbar_info['location']
    anchor = cbax._colorbar_info['anchor']
    fraction = cbax._colorbar_info['fraction']
    aspect = cbax._colorbar_info['aspect']
    shrink = cbax._colorbar_info['shrink']
    cbpos, cbbbox = get_pos_and_bbox(cbax, renderer)
    cbpad = colorbar_get_pad(layoutgrids, cbax)
    if location in ('left', 'right'):
        pbcb = pb.shrunk(fraction, shrink).anchored(anchor, pb)
        if location == 'right':
            lmargin = cbpos.x0 - cbbbox.x0
            dx = bboxparent.x1 - pbcb.x0 + offset['right']
            dx += cbpad + lmargin
            offset['right'] += cbbbox.width + cbpad
            pbcb = pbcb.translated(dx, 0)
        else:
            lmargin = cbpos.x0 - cbbbox.x0
            dx = bboxparent.x0 - pbcb.x0
            dx += -cbbbox.width - cbpad + lmargin - offset['left']
            offset['left'] += cbbbox.width + cbpad
            pbcb = pbcb.translated(dx, 0)
    else:
        pbcb = pb.shrunk(shrink, fraction).anchored(anchor, pb)
        if location == 'top':
            bmargin = cbpos.y0 - cbbbox.y0
            dy = bboxparent.y1 - pbcb.y0 + offset['top']
            dy += cbpad + bmargin
            offset['top'] += cbbbox.height + cbpad
            pbcb = pbcb.translated(0, dy)
        else:
            bmargin = cbpos.y0 - cbbbox.y0
            dy = bboxparent.y0 - pbcb.y0
            dy += -cbbbox.height - cbpad + bmargin - offset['bottom']
            offset['bottom'] += cbbbox.height + cbpad
            pbcb = pbcb.translated(0, dy)
    pbcb = trans_fig_to_subfig.transform_bbox(pbcb)
    cbax.set_transform(fig.transSubfigure)
    cbax._set_position(pbcb)
    cbax.set_anchor(anchor)
    if location in ['bottom', 'top']:
        aspect = 1 / aspect
    cbax.set_box_aspect(aspect)
    cbax.set_aspect('auto')
    return offset