from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
from .axisline_style import AxislineStyle
def get_window_extents(self, renderer=None):
    if renderer is None:
        renderer = self.figure._get_renderer()
    if not self.get_visible():
        self._axislabel_pad = self._external_pad
        return []
    bboxes = []
    r, total_width = self._get_ticklabels_offsets(renderer, self._axis_direction)
    pad = self._external_pad + renderer.points_to_pixels(self.get_pad())
    self._offset_radius = r + pad
    for (x, y), a, l in self._locs_angles_labels:
        self._ref_angle = a
        self.set_x(x)
        self.set_y(y)
        self.set_text(l)
        bb = LabelBase.get_window_extent(self, renderer)
        bboxes.append(bb)
    self._axislabel_pad = total_width + pad
    return bboxes