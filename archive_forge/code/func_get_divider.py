from numbers import Number
import functools
from types import MethodType
import numpy as np
from matplotlib import _api, cbook
from matplotlib.gridspec import SubplotSpec
from .axes_divider import Size, SubplotDivider, Divider
from .mpl_axes import Axes, SimpleAxisArtist
def get_divider(self):
    return self._divider