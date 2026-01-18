import logging
import numpy as np
from matplotlib import _api, artist as martist
import matplotlib.transforms as mtransforms
import matplotlib._layoutgrid as mlayoutgrid
def make_margin_suptitles(layoutgrids, fig, renderer, *, w_pad=0, h_pad=0):
    inv_trans_fig = fig.transFigure.inverted().transform_bbox
    padbox = mtransforms.Bbox([[0, 0], [w_pad, h_pad]])
    padbox = (fig.transFigure - fig.transSubfigure).transform_bbox(padbox)
    h_pad_local = padbox.height
    w_pad_local = padbox.width
    for sfig in fig.subfigs:
        make_margin_suptitles(layoutgrids, sfig, renderer, w_pad=w_pad, h_pad=h_pad)
    if fig._suptitle is not None and fig._suptitle.get_in_layout():
        p = fig._suptitle.get_position()
        if getattr(fig._suptitle, '_autopos', False):
            fig._suptitle.set_position((p[0], 1 - h_pad_local))
            bbox = inv_trans_fig(fig._suptitle.get_tightbbox(renderer))
            layoutgrids[fig].edit_margin_min('top', bbox.height + 2 * h_pad)
    if fig._supxlabel is not None and fig._supxlabel.get_in_layout():
        p = fig._supxlabel.get_position()
        if getattr(fig._supxlabel, '_autopos', False):
            fig._supxlabel.set_position((p[0], h_pad_local))
            bbox = inv_trans_fig(fig._supxlabel.get_tightbbox(renderer))
            layoutgrids[fig].edit_margin_min('bottom', bbox.height + 2 * h_pad)
    if fig._supylabel is not None and fig._supylabel.get_in_layout():
        p = fig._supylabel.get_position()
        if getattr(fig._supylabel, '_autopos', False):
            fig._supylabel.set_position((w_pad_local, p[1]))
            bbox = inv_trans_fig(fig._supylabel.get_tightbbox(renderer))
            layoutgrids[fig].edit_margin_min('left', bbox.width + 2 * w_pad)