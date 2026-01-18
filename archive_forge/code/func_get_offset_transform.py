import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def get_offset_transform(self):
    """Return the `.Transform` instance used by this artist offset."""
    if self._offset_transform is None:
        self._offset_transform = transforms.IdentityTransform()
    elif not isinstance(self._offset_transform, transforms.Transform) and hasattr(self._offset_transform, '_as_mpl_transform'):
        self._offset_transform = self._offset_transform._as_mpl_transform(self.axes)
    return self._offset_transform