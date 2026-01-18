import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_segments(self, segments):
    if segments is None:
        return
    self._paths = [mpath.Path(seg) if isinstance(seg, np.ma.MaskedArray) else mpath.Path(np.asarray(seg, float)) for seg in segments]
    self.stale = True