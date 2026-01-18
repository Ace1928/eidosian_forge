from __future__ import annotations
from typing import TYPE_CHECKING
import logging # isort:skip
import numpy as np
from ..core.enums import HorizontalLocation, MarkerType, VerticalLocation
from ..core.properties import (
from ..models import (
from ..models.dom import Template
from ..models.tools import (
from ..transform import linear_cmap
from ..util.options import Options
from ._graph import get_graph_kwargs
from ._plot import get_range, get_scale, process_axis_and_grid
from ._stack import double_stack, single_stack
from ._tools import process_active_tools, process_tools_arg
from .contour import ContourRenderer, from_contour
from .glyph_api import _MARKER_SHORTCUTS, GlyphAPI
def markers():
    """ Prints a list of valid marker types for scatter()

    Returns:
        None
    """
    print('Available markers: \n\n - ' + '\n - '.join(list(MarkerType)))
    print()
    print('Shortcuts: \n\n' + '\n'.join((f' {short!r}: {name}' for short, name in _MARKER_SHORTCUTS.items())))