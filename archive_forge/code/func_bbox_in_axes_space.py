from __future__ import annotations
import typing
from matplotlib.transforms import Affine2D, Bbox
from .transforms import ZEROS_BBOX
def bbox_in_axes_space(artist: Artist, ax: Axes, renderer: RendererBase) -> Bbox:
    """
    Bounding box of artist in figure coordinates
    """
    box = artist.get_window_extent(renderer) or ZEROS_BBOX
    return ax.transAxes.inverted().transform_bbox(box)