from __future__ import annotations
import typing
from matplotlib.transforms import Affine2D, Bbox
from .transforms import ZEROS_BBOX
def pts_in_figure_space(fig: Figure, pts: float) -> float:
    """
    Points in figure coordinates
    """
    return fig.transFigure.inverted().transform([0, pts])[1]