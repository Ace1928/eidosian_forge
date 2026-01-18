from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
def apply_aspect(self, position=None):
    if position is None:
        position = self.get_position(original=True)
    trans = self.get_figure().transSubfigure
    bb = mtransforms.Bbox.unit().transformed(trans)
    fig_aspect = bb.height / bb.width
    box_aspect = 1
    pb = position.frozen()
    pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect)
    self._set_position(pb1.anchored(self.get_anchor(), pb), 'active')