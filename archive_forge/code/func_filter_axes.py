from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def filter_axes(axs: list[Axes], get: AxesLocation='all') -> list[Axes]:
    """
    Return subset of axes
    """
    if get == 'all':
        return axs
    pred_method = f'is_{get}'
    return [ax for spec, ax in zip(get_subplotspec_list(axs), axs) if getattr(spec, pred_method)()]