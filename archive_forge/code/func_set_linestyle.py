import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_linestyle(self, ls):
    """
        Set the linestyle(s) for the collection.

        ===========================   =================
        linestyle                     description
        ===========================   =================
        ``'-'`` or ``'solid'``        solid line
        ``'--'`` or  ``'dashed'``     dashed line
        ``'-.'`` or  ``'dashdot'``    dash-dotted line
        ``':'`` or ``'dotted'``       dotted line
        ===========================   =================

        Alternatively a dash tuple of the following form can be provided::

            (offset, onoffseq),

        where ``onoffseq`` is an even length tuple of on and off ink in points.

        Parameters
        ----------
        ls : str or tuple or list thereof
            Valid values for individual linestyles include {'-', '--', '-.',
            ':', '', (offset, on-off-seq)}. See `.Line2D.set_linestyle` for a
            complete description.
        """
    try:
        dashes = [mlines._get_dash_pattern(ls)]
    except ValueError:
        try:
            dashes = [mlines._get_dash_pattern(x) for x in ls]
        except ValueError as err:
            emsg = f'Do not know how to convert {ls!r} to dashes'
            raise ValueError(emsg) from err
    self._us_linestyles = dashes
    self._linewidths, self._linestyles = self._bcast_lwls(self._us_lw, self._us_linestyles)