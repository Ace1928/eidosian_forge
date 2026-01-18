from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def get_yaxis_tick_pads(pack: LayoutPack, ax: Axes) -> Iterator[float]:
    """
    Return YTicks paddings
    """
    is_blank = pack.theme.T.is_blank
    major, minor = ([], [])
    if not is_blank('axis_text_y'):
        w = pack.figure.get_figwidth() * 72
        major = [(t.get_pad() or 0) / w for t in ax.yaxis.get_major_ticks()]
        minor = [(t.get_pad() or 0) / w for t in ax.yaxis.get_minor_ticks()]
    return chain(major, minor)