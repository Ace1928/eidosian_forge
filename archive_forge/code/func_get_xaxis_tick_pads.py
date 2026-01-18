from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def get_xaxis_tick_pads(pack: LayoutPack, ax: Axes) -> Iterator[float]:
    """
    Return XTicks paddings
    """
    is_blank = pack.theme.T.is_blank
    major, minor = ([], [])
    if not is_blank('axis_text_y'):
        h = pack.figure.get_figheight() * 72
        major = [(t.get_pad() or 0) / h for t in ax.xaxis.get_major_ticks()]
        minor = [(t.get_pad() or 0) / h for t in ax.xaxis.get_minor_ticks()]
    return chain(major, minor)