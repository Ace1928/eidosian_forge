from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import (
from .palettes import color_palette, blend_palette
from ._docstrings import (
def map_offdiag(self, func, **kwargs):
    """Plot with a bivariate function on the off-diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
    if self.square_grid:
        self.map_lower(func, **kwargs)
        if not self._corner:
            self.map_upper(func, **kwargs)
    else:
        indices = []
        for i, y_var in enumerate(self.y_vars):
            for j, x_var in enumerate(self.x_vars):
                if x_var != y_var:
                    indices.append((i, j))
        self._map_bivariate(func, indices, **kwargs)
    return self