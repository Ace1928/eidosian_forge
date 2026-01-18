from __future__ import annotations
from matplotlib.offsetbox import (
from matplotlib.patches import bbox_artist as mbbox_artist
from matplotlib.transforms import Affine2D, Bbox
from .patches import InsideStrokedRectangle
def get_bbox(self, renderer):
    self._correct_dpi(renderer)
    return super().get_bbox(renderer)