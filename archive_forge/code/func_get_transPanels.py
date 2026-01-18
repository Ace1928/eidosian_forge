from __future__ import annotations
import typing
from matplotlib.transforms import Affine2D, Bbox
from .transforms import ZEROS_BBOX
def get_transPanels(fig: Figure) -> Transform:
    """
    Coordinate system of the Panels (facets) area

    (0, 0) is the bottom-left of the bottom-left panel and
    (1, 1) is the top right of the top-right panel.

    The subplot parameters must be set before calling this function.
    i.e. fig.subplots_adjust should have been called.
    """
    params = fig.subplotpars
    W, H = (fig.bbox.width, fig.bbox.height)
    sx, sy = (params.right - params.left, params.top - params.bottom)
    dx, dy = (params.left * W, params.bottom * H)
    transFiguretoPanels = Affine2D().scale(sx, sy).translate(dx, dy)
    return fig.transFigure + transFiguretoPanels