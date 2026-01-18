import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
def new_horizontal(self, size, pad=None, pack_start=False, **kwargs):
    """
        Helper method for ``append_axes("left")`` and ``append_axes("right")``.

        See the documentation of `append_axes` for more details.

        :meta private:
        """
    if pad is None:
        pad = mpl.rcParams['figure.subplot.wspace'] * self._xref
    pos = 'left' if pack_start else 'right'
    if pad:
        if not isinstance(pad, Size._Base):
            pad = Size.from_any(pad, fraction_ref=self._xref)
        self.append_size(pos, pad)
    if not isinstance(size, Size._Base):
        size = Size.from_any(size, fraction_ref=self._xref)
    self.append_size(pos, size)
    locator = self.new_locator(nx=0 if pack_start else len(self._horizontal) - 1, ny=self._yrefindex)
    ax = self._get_new_axes(**kwargs)
    ax.set_axes_locator(locator)
    return ax