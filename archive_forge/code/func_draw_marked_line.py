import warnings
import itertools
from contextlib import contextmanager
from packaging.version import Version
import numpy as np
import matplotlib as mpl
from matplotlib import transforms
from .. import utils
def draw_marked_line(self, data, coordinates, linestyle, markerstyle, label, mplobj=None):
    """Draw a line that also has markers.

        If this isn't reimplemented by a renderer object, by default, it will
        make a call to BOTH draw_line and draw_markers when both markerstyle
        and linestyle are not None in the same Line2D object.

        """
    if linestyle is not None:
        self.draw_line(data, coordinates, linestyle, label, mplobj)
    if markerstyle is not None:
        self.draw_markers(data, coordinates, markerstyle, label, mplobj)