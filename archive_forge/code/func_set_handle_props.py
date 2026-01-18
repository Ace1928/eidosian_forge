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
def set_handle_props(self, **handle_props):
    """
        Set the properties of the handles selector artist. See the
        `handle_props` argument in the selector docstring to know which
        properties are supported.
        """
    if not hasattr(self, '_handles_artists'):
        raise NotImplementedError("This selector doesn't have handles.")
    artist = self._handles_artists[0]
    handle_props = cbook.normalize_kwargs(handle_props, artist)
    for handle in self._handles_artists:
        handle.set(**handle_props)
    if self.useblit:
        self.update()
    self._handle_props.update(handle_props)