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
def get_tight_bbox(fig, bbox_extra_artists=None, pad=None):
    """
    Compute a tight bounding box around all the artists in the figure.
    """
    if bbox_extra_artists is None:
        bbox_extra_artists = []
    renderer = fig.canvas.get_renderer()
    bbox_inches = fig.get_tightbbox(renderer)
    bbox_artists = bbox_extra_artists[:]
    bbox_artists += fig.get_default_bbox_extra_artists()
    bbox_filtered = []
    for a in bbox_artists:
        bbox = a.get_window_extent(renderer)
        if isinstance(bbox, tuple):
            continue
        if a.get_clip_on():
            clip_box = a.get_clip_box()
            if clip_box is not None:
                bbox = Bbox.intersection(bbox, clip_box)
            clip_path = a.get_clip_path()
            if clip_path is not None and bbox is not None:
                clip_path = clip_path.get_fully_transformed_path()
                bbox = Bbox.intersection(bbox, clip_path.get_extents())
        if bbox is not None and (bbox.width != 0 or bbox.height != 0) and np.isfinite(bbox).all():
            bbox_filtered.append(bbox)
    if bbox_filtered:
        _bbox = Bbox.union(bbox_filtered)
        trans = Affine2D().scale(1.0 / fig.dpi)
        bbox_extra = TransformedBbox(_bbox, trans)
        bbox_inches = Bbox.union([bbox_inches, bbox_extra])
    return bbox_inches.padded(pad) if pad else bbox_inches