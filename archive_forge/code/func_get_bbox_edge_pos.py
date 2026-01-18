from matplotlib import _api, _docstring
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxTransformTo
from matplotlib.transforms import IdentityTransform, TransformedBbox
from . import axes_size as Size
from .parasite_axes import HostAxes
@staticmethod
def get_bbox_edge_pos(bbox, loc):
    """
        Return the ``(x, y)`` coordinates of corner *loc* of *bbox*; parameters
        behave as documented for the `.BboxConnector` constructor.
        """
    x0, y0, x1, y1 = bbox.extents
    if loc == 1:
        return (x1, y1)
    elif loc == 2:
        return (x0, y1)
    elif loc == 3:
        return (x0, y0)
    elif loc == 4:
        return (x1, y0)