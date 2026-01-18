import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
def new_vertical(self, size, pad=None, pack_start=False, **kwargs):
    """
        Helper method for ``append_axes("top")`` and ``append_axes("bottom")``.

        See the documentation of `append_axes` for more details.

        :meta private:
        """
    if pad is None:
        pad = mpl.rcParams['figure.subplot.hspace'] * self._yref
    pos = 'bottom' if pack_start else 'top'
    if pad:
        if not isinstance(pad, Size._Base):
            pad = Size.from_any(pad, fraction_ref=self._yref)
        self.append_size(pos, pad)
    if not isinstance(size, Size._Base):
        size = Size.from_any(size, fraction_ref=self._yref)
    self.append_size(pos, size)
    locator = self.new_locator(nx=self._xrefindex, ny=0 if pack_start else len(self._vertical) - 1)
    ax = self._get_new_axes(**kwargs)
    ax.set_axes_locator(locator)
    return ax