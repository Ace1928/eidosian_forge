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
def get_old_rcparams():
    deprecated_rcparams = ['text.latex.unicode', 'examples.directory', 'savefig.frameon', 'verbose.level', 'verbose.fileo', 'datapath', 'text.latex.preview', 'animation.avconv_args', 'animation.avconv_path', 'animation.html_args', 'keymap.all_axes', 'savefig.jpeg_quality']
    old_rcparams = {k: v for k, v in mpl.rcParams.items() if mpl_version < Version('3.0') or k not in deprecated_rcparams}
    return old_rcparams