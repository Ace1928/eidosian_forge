import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
import matplotlib.artist as martist
import matplotlib.path as mpath
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import (
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
@_api.deprecated('3.7', alternative='get_bbox and child.get_offset')
def get_extent_offsets(self, renderer):
    """
        Update offset of the children and return the extent of the box.

        Parameters
        ----------
        renderer : `.RendererBase` subclass

        Returns
        -------
        width
        height
        xdescent
        ydescent
        list of (xoffset, yoffset) pairs
        """
    bbox, offsets = self._get_bbox_and_child_offsets(renderer)
    return (bbox.width, bbox.height, -bbox.x0, -bbox.y0, offsets)