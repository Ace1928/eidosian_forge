from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def max_ylabels_width(pack: LayoutPack, axes_loc: AxesLocation='all') -> float:
    """
    Return maximum width[inches] of y tick labels
    """
    widths = [tight_bbox_in_figure_space(label, pack.figure, pack.renderer).width + pad for ax in filter_axes(pack.axs, axes_loc) for label, pad in zip(get_yaxis_labels(pack, ax), get_yaxis_tick_pads(pack, ax))]
    return max(widths) if len(widths) else 0