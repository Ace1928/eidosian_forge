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
def sharez(self, other):
    """
        Share the z-axis with *other*.

        This is equivalent to passing ``sharez=other`` when constructing the
        Axes, and cannot be used if the z-axis is already being shared with
        another Axes.
        """
    _api.check_isinstance(Axes3D, other=other)
    if self._sharez is not None and other is not self._sharez:
        raise ValueError('z-axis is already shared')
    self._shared_axes['z'].join(self, other)
    self._sharez = other
    self.zaxis.major = other.zaxis.major
    self.zaxis.minor = other.zaxis.minor
    z0, z1 = other.get_zlim()
    self.set_zlim(z0, z1, emit=False, auto=other.get_autoscalez_on())
    self.zaxis._scale = other.zaxis._scale