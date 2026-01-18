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
def set_proj_type(self, proj_type, focal_length=None):
    """
        Set the projection type.

        Parameters
        ----------
        proj_type : {'persp', 'ortho'}
            The projection type.
        focal_length : float, default: None
            For a projection type of 'persp', the focal length of the virtual
            camera. Must be > 0. If None, defaults to 1.
            The focal length can be computed from a desired Field Of View via
            the equation: focal_length = 1/tan(FOV/2)
        """
    _api.check_in_list(['persp', 'ortho'], proj_type=proj_type)
    if proj_type == 'persp':
        if focal_length is None:
            focal_length = 1
        elif focal_length <= 0:
            raise ValueError(f'focal_length = {focal_length} must be greater than 0')
        self._focal_length = focal_length
    else:
        if focal_length not in (None, np.inf):
            raise ValueError(f'focal_length = {focal_length} must be None for proj_type = {proj_type}')
        self._focal_length = np.inf