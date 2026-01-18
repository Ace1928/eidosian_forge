from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def max_xlabels_right_protrusion(pack: LayoutPack, axes_loc: AxesLocation='all') -> float:
    """
    Return maximum width[inches] of x tick labels to the right of the axes
    """

    def get_artist_right_x(a: Artist) -> float:
        xy = bbox_in_figure_space(a, pack.figure, pack.renderer).max
        return xy[0]
    extras = []
    for ax in filter_axes(pack.axs, axes_loc):
        ax_right = get_artist_right_x(ax)
        for label in get_xaxis_labels(pack, ax):
            label_right = get_artist_right_x(label)
            extras.append(max(0, label_right - ax_right))
    return max(extras) if len(extras) else 0