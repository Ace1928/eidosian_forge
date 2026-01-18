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
def set_radio_props(self, props):
    """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the radio
            buttons.
        """
    _api.check_isinstance(dict, props=props)
    if 's' in props:
        props['sizes'] = np.broadcast_to(props.pop('s'), len(self.labels))
    self._buttons.update(props)
    self._active_colors = self._buttons.get_facecolor()
    if len(self._active_colors) == 1:
        self._active_colors = np.repeat(self._active_colors, len(self.labels), axis=0)
    self._buttons.set_facecolor([activecolor if text.get_text() == self.value_selected else 'none' for text, activecolor in zip(self.labels, self._active_colors)])