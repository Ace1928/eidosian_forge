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
def polygons_to_path_patches(element):
    """
    Converts Polygons into list of lists of matplotlib.patches.PathPatch
    objects including any specified holes. Each list represents one
    (multi-)polygon.
    """
    paths = element.split(datatype='array', dimensions=element.kdims)
    has_holes = isinstance(element, Polygons) and element.interface.has_holes(element)
    holes = element.interface.holes(element) if has_holes else None
    mpl_paths = []
    for i, path in enumerate(paths):
        splits = np.where(np.isnan(path[:, :2].astype('float')).sum(axis=1))[0]
        arrays = np.split(path, splits + 1) if len(splits) else [path]
        subpath = []
        for j, array in enumerate(arrays):
            if j != len(arrays) - 1:
                array = array[:-1]
            if (array[0] != array[-1]).any():
                array = np.append(array, array[:1], axis=0)
            interiors = []
            for interior in holes[i][j] if has_holes else []:
                if (interior[0] != interior[-1]).any():
                    interior = np.append(interior, interior[:1], axis=0)
                interiors.append(interior)
            vertices = np.concatenate([array] + interiors)
            codes = np.concatenate([ring_coding(array)] + [ring_coding(h) for h in interiors])
            subpath.append(PathPatch(Path(vertices, codes)))
        mpl_paths.append(subpath)
    return mpl_paths