from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
@_api.deprecated('3.7', addendum='Any custom property styling may be lost.')
@property
def circles(self):
    if not hasattr(self, '_circles'):
        radius = min(0.5 / (len(self.labels) + 1) - 0.01, 0.05)
        circles = self._circles = [Circle(xy=self._buttons.get_offsets()[i], edgecolor='black', facecolor=self._buttons.get_facecolor()[i], radius=radius, transform=self.ax.transAxes, animated=self._useblit) for i in range(len(self.labels))]
        self._buttons.set_visible(False)
        for circle in circles:
            self.ax.add_patch(circle)
    return self._circles