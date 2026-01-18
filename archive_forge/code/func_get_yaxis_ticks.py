from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def get_yaxis_ticks(pack: LayoutPack, ax: Axes) -> Iterator[Tick]:
    """
    Return all YTicks that will be shown
    """
    is_blank = pack.theme.T.is_blank
    major, minor = ([], [])
    if not is_blank('axis_ticks_major_y'):
        major = ax.yaxis.get_major_ticks()
    if not is_blank('axis_ticks_minor_y'):
        minor = ax.yaxis.get_minor_ticks()
    return chain(major, minor)