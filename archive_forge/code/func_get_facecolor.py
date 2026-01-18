import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def get_facecolor(self):
    fc = super().get_facecolor()
    unmasked_polys = self._get_unmasked_polys().ravel()
    if len(fc) != len(unmasked_polys):
        return fc
    return fc[unmasked_polys, :]