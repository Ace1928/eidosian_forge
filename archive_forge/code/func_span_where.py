import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
@classmethod
@_api.deprecated('3.7', alternative='fill_between')
def span_where(cls, x, ymin, ymax, where, **kwargs):
    """
        Return a `.BrokenBarHCollection` that plots horizontal bars from
        over the regions in *x* where *where* is True.  The bars range
        on the y-axis from *ymin* to *ymax*

        *kwargs* are passed on to the collection.
        """
    xranges = []
    for ind0, ind1 in cbook.contiguous_regions(where):
        xslice = x[ind0:ind1]
        if not len(xslice):
            continue
        xranges.append((xslice[0], xslice[-1] - xslice[0]))
    return BrokenBarHCollection(xranges, [ymin, ymax - ymin], **kwargs)