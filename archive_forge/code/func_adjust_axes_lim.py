import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.patches as mpatches
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.parasite_axes import host_axes_class_factory
from . import axislines, grid_helper_curvelinear
from .axis_artist import AxisArtist
from .grid_finder import ExtremeFinderSimple
def adjust_axes_lim(self):
    bbox = self.patch.get_path().get_extents(self.patch.get_transform() - self.transData)
    bbox = bbox.expanded(1.02, 1.02)
    self.set_xlim(bbox.xmin, bbox.xmax)
    self.set_ylim(bbox.ymin, bbox.ymax)