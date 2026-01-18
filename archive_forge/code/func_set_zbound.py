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
def set_zbound(self, lower=None, upper=None):
    """
        Set the lower and upper numerical bounds of the z-axis.

        This method will honor axes inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscalez_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.

        See Also
        --------
        get_zbound
        get_zlim, set_zlim
        invert_zaxis, zaxis_inverted
        """
    if upper is None and np.iterable(lower):
        lower, upper = lower
    old_lower, old_upper = self.get_zbound()
    if lower is None:
        lower = old_lower
    if upper is None:
        upper = old_upper
    self.set_zlim(sorted((lower, upper), reverse=bool(self.zaxis_inverted())), auto=None)