from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
def autofmt_xdate(self, bottom=0.2, rotation=30, ha='right', which='major'):
    """
        Date ticklabels often overlap, so it is useful to rotate them
        and right align them.  Also, a common use case is a number of
        subplots with shared x-axis where the x-axis is date data.  The
        ticklabels are often long, and it helps to rotate them on the
        bottom subplot and turn them off on other subplots, as well as
        turn off xlabels.

        Parameters
        ----------
        bottom : float, default: 0.2
            The bottom of the subplots for `subplots_adjust`.
        rotation : float, default: 30 degrees
            The rotation angle of the xtick labels in degrees.
        ha : {'left', 'center', 'right'}, default: 'right'
            The horizontal alignment of the xticklabels.
        which : {'major', 'minor', 'both'}, default: 'major'
            Selects which ticklabels to rotate.
        """
    _api.check_in_list(['major', 'minor', 'both'], which=which)
    allsubplots = all((ax.get_subplotspec() for ax in self.axes))
    if len(self.axes) == 1:
        for label in self.axes[0].get_xticklabels(which=which):
            label.set_ha(ha)
            label.set_rotation(rotation)
    elif allsubplots:
        for ax in self.get_axes():
            if ax.get_subplotspec().is_last_row():
                for label in ax.get_xticklabels(which=which):
                    label.set_ha(ha)
                    label.set_rotation(rotation)
            else:
                for label in ax.get_xticklabels(which=which):
                    label.set_visible(False)
                ax.set_xlabel('')
    if allsubplots:
        self.subplots_adjust(bottom=bottom)
    self.stale = True