from __future__ import annotations
import typing
from copy import deepcopy
from dataclasses import dataclass
from ._plot_side_space import LRTBSpaces, WHSpaceParts, calculate_panel_spacing
from .utils import bbox_in_figure_space, get_transPanels
def set_figure_artist_positions(pack: LayoutPack, tparams: TightParams):
    """
    Set the x,y position of the artists around the panels
    """
    theme = pack.theme
    sides = tparams.sides
    params = tparams.params
    if pack.plot_title:
        ha = theme.getp(('plot_title', 'ha'))
        pack.plot_title.set_y(sides.t.edge('plot_title'))
        horizonally_align_text_with_panels(pack.plot_title, params, ha, pack)
    if pack.plot_subtitle:
        ha = theme.getp(('plot_subtitle', 'ha'))
        pack.plot_subtitle.set_y(sides.t.edge('plot_subtitle'))
        horizonally_align_text_with_panels(pack.plot_subtitle, params, ha, pack)
    if pack.plot_caption:
        ha = theme.getp(('plot_caption', 'ha'), 'right')
        pack.plot_caption.set_y(sides.b.edge('plot_caption'))
        horizonally_align_text_with_panels(pack.plot_caption, params, ha, pack)
    if pack.axis_title_x:
        ha = theme.getp(('axis_title_x', 'ha'), 'center')
        pack.axis_title_x.set_y(sides.b.edge('axis_title_x'))
        horizonally_align_text_with_panels(pack.axis_title_x, params, ha, pack)
    if pack.axis_title_y:
        va = theme.getp(('axis_title_y', 'va'), 'center')
        pack.axis_title_y.set_x(sides.l.edge('axis_title_y'))
        vertically_align_text_with_panels(pack.axis_title_y, params, va, pack)
    if pack.legends:
        set_legends_position(pack.legends, tparams, pack.figure)