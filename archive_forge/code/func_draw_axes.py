import warnings
import itertools
from contextlib import contextmanager
from packaging.version import Version
import numpy as np
import matplotlib as mpl
from matplotlib import transforms
from .. import utils
@contextmanager
def draw_axes(self, ax, props):
    if hasattr(self, '_current_ax') and self._current_ax is not None:
        warnings.warn('axes embedded in axes: something is wrong')
    self._current_ax = ax
    self._ax_props = props
    self.open_axes(ax=ax, props=props)
    yield
    self.close_axes(ax=ax)
    self._current_ax = None
    self._ax_props = {}