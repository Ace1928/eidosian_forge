from __future__ import annotations
import typing
from matplotlib.transforms import Affine2D, Bbox
from .transforms import ZEROS_BBOX
def bbox_in_figure_space(artist: Artist, fig: Figure, renderer: RendererBase) -> Bbox:
    """
    Bounding box of artist in figure coordinates
    """
    box = artist.get_window_extent(renderer) or ZEROS_BBOX
    return fig.transFigure.inverted().transform_bbox(box)