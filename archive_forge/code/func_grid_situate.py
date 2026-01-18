from contextlib import contextmanager
from itertools import chain
import matplotlib as mpl
import numpy as np
import param
from matplotlib import (
from matplotlib.font_manager import font_scalings
from mpl_toolkits.mplot3d import Axes3D  # noqa (For 3D plots)
from ...core import (
from ...core.options import SkipRendering, Store
from ...core.util import int_to_alpha, int_to_roman, wrap_tuple_streams
from ..plot import (
from ..util import attach_streams, collate, displayable
from .util import compute_ratios, fix_aspect, get_old_rcparams
def grid_situate(self, current_idx, layout_type, subgrid_width):
    """
        Situate the current AdjointLayoutPlot in a LayoutPlot. The
        LayoutPlot specifies a layout_type into which the AdjointLayoutPlot
        must be embedded. This enclosing layout is guaranteed to have
        enough cells to display all the views.

        Based on this enforced layout format, a starting index
        supplied by LayoutPlot (indexing into a large gridspec
        arrangement) is updated to the appropriate embedded value. It
        will also return a list of gridspec indices associated with
        the all the required layout axes.
        """
    if layout_type == 'Single':
        start, inds = (current_idx + 1, [current_idx])
    elif layout_type == 'Dual':
        start, inds = (current_idx + 2, [current_idx, current_idx + 1])
    bottom_idx = current_idx + subgrid_width
    if layout_type == 'Embedded Dual':
        bottom = (current_idx + 1) % subgrid_width == 0
        grid_idx = (bottom_idx if bottom else current_idx) + 1
        start, inds = (grid_idx, [current_idx, bottom_idx])
    elif layout_type == 'Triple':
        bottom = (current_idx + 2) % subgrid_width == 0
        grid_idx = (bottom_idx if bottom else current_idx) + 2
        start, inds = (grid_idx, [current_idx, current_idx + 1, bottom_idx, bottom_idx + 1])
    return (start, inds)