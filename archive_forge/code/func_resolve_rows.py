import inspect
import re
import warnings
import matplotlib as mpl
import numpy as np
from matplotlib import (
from matplotlib.colors import Normalize, cnames
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Path, PathPatch
from matplotlib.rcsetup import validate_fontsize, validate_fonttype, validate_hatch
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from packaging.version import Version
from ...core.util import arraylike_types, cftime_types, is_number
from ...element import RGB, Polygons, Raster
from ..util import COLOR_ALIASES, RGB_HEX_REGEX
def resolve_rows(rows):
    """
    Recursively iterate over lists of axes merging
    them by their vertical overlap leaving a list
    of rows.
    """
    merged_rows = []
    for row in rows:
        overlap = False
        for mrow in merged_rows:
            if any((axis_overlap(ax1, ax2) for ax1 in row for ax2 in mrow)):
                mrow += row
                overlap = True
                break
        if not overlap:
            merged_rows.append(row)
    if rows == merged_rows:
        return rows
    else:
        return resolve_rows(merged_rows)