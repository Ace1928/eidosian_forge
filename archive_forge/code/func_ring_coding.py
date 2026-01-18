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
def ring_coding(array):
    """
    Produces matplotlib Path codes for exterior and interior rings
    of a polygon geometry.
    """
    n = len(array)
    codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    return codes