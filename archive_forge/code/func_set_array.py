import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
def set_array(self, A):
    prev_unmask = self._get_unmasked_polys()
    if self._deprecated_compression and np.ndim(A) == 1:
        _api.warn_deprecated('3.8', message=f'Setting a PolyQuadMesh array using the compressed values is deprecated. Pass the full 2D shape of the original array {prev_unmask.shape} including the masked elements.')
        Afull = np.empty(self._original_mask.shape)
        Afull[~self._original_mask] = A
        mask = self._original_mask.copy()
        mask[~self._original_mask] |= np.ma.getmask(A)
        A = np.ma.array(Afull, mask=mask)
        return super().set_array(A)
    self._deprecated_compression = False
    super().set_array(A)
    if not np.array_equal(prev_unmask, self._get_unmasked_polys()):
        self._set_unmasked_verts()