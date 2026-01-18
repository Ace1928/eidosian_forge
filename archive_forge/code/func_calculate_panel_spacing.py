from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def calculate_panel_spacing(pack: LayoutPack, spaces: LRTBSpaces) -> WHSpaceParts:
    """
    Spacing between the panels (wspace & hspace)

    Both spaces are calculated from a fraction of the width.
    This ensures that the same fraction gives equals space
    in both directions.
    """
    if isinstance(pack.facet, facet_wrap):
        return _calculate_panel_spacing_facet_wrap(pack, spaces)
    elif isinstance(pack.facet, facet_grid):
        return _calculate_panel_spacing_facet_grid(pack, spaces)
    elif isinstance(pack.facet, facet_null):
        return _calculate_panel_spacing_facet_null(pack, spaces)
    return WHSpaceParts(0, 0, 0, 0, 0, 0, 0, 0)