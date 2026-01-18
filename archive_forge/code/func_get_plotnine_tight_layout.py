from __future__ import annotations
import typing
from copy import deepcopy
from dataclasses import dataclass
from ._plot_side_space import LRTBSpaces, WHSpaceParts, calculate_panel_spacing
from .utils import bbox_in_figure_space, get_transPanels
def get_plotnine_tight_layout(pack: LayoutPack) -> TightParams:
    """
    Compute tight layout parameters
    """
    sides = LRTBSpaces(pack)
    gullies = calculate_panel_spacing(pack, sides)
    tight_params = TightParams(sides, gullies)
    ratio = pack.facet._aspect_ratio()
    if ratio is not None:
        tight_params = tight_params.to_aspect_ratio(pack.facet, ratio, gullies)
    return tight_params