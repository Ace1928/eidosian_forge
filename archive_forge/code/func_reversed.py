import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def reversed(self, name=None):
    """
        Return a reversed instance of the Colormap.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        Returns
        -------
        ListedColormap
            A reversed instance of the colormap.
        """
    if name is None:
        name = self.name + '_r'
    colors_r = list(reversed(self.colors))
    new_cmap = ListedColormap(colors_r, name=name, N=self.N)
    new_cmap._rgba_over = self._rgba_under
    new_cmap._rgba_under = self._rgba_over
    new_cmap._rgba_bad = self._rgba_bad
    return new_cmap