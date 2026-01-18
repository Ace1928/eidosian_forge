from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def max_ylabels_top_protrusion(pack: LayoutPack, axes_loc: AxesLocation='all') -> float:
    """
    Return maximum height[inches] above the axes of y tick labels
    """

    def get_artist_top_y(a: Artist) -> float:
        xy = bbox_in_figure_space(a, pack.figure, pack.renderer).max
        return xy[1]
    extras = []
    for ax in filter_axes(pack.axs, axes_loc):
        ax_top = get_artist_top_y(ax)
        for label in get_yaxis_labels(pack, ax):
            label_top = get_artist_top_y(label)
            extras.append(max(0, label_top - ax_top))
    return max(extras) if len(extras) else 0