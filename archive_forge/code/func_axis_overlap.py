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
def axis_overlap(ax1, ax2):
    """
    Tests whether two axes overlap vertically
    """
    b1, t1 = ax1.get_position().intervaly
    b2, t2 = ax2.get_position().intervaly
    return t1 > b2 and b1 < t2