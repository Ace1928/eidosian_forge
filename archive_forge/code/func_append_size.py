import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
def append_size(self, position, size):
    _api.check_in_list(['left', 'right', 'bottom', 'top'], position=position)
    if position == 'left':
        self._horizontal.insert(0, size)
        self._xrefindex += 1
    elif position == 'right':
        self._horizontal.append(size)
    elif position == 'bottom':
        self._vertical.insert(0, size)
        self._yrefindex += 1
    else:
        self._vertical.append(size)