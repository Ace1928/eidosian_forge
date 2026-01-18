from __future__ import annotations
import typing
from copy import deepcopy
from dataclasses import dataclass
from ._plot_side_space import LRTBSpaces, WHSpaceParts, calculate_panel_spacing
from .utils import bbox_in_figure_space, get_transPanels
def horizonally_align_text_with_panels(text: Text, params: GridSpecParams, ha: str | float, pack: LayoutPack):
    """
    Horizontal justification

    Reinterpret horizontal alignment to be justification about the panels.
    """
    if isinstance(ha, str):
        lookup = {'left': 0.0, 'center': 0.5, 'right': 1.0}
        f = lookup[ha]
    else:
        f = ha
    box = bbox_in_figure_space(text, pack.figure, pack.renderer)
    x = params.left * (1 - f) + (params.right - box.width) * f
    text.set_x(x)
    text.set_horizontalalignment('left')