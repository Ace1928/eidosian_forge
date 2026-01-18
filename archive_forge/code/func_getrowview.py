from bisect import bisect_left
import numpy as np
from ._matrix import spmatrix
from ._base import _spbase, sparray, issparse
from ._index import IndexMixin, INT_TYPES, _broadcast_arrays
from ._sputils import (getdtype, isshape, isscalarlike, upcast_scalar,
from . import _csparsetools
def getrowview(self, i):
    """Returns a view of the 'i'th row (without copying).
        """
    new = self._lil_container((1, self.shape[1]), dtype=self.dtype)
    new.rows[0] = self.rows[i]
    new.data[0] = self.data[i]
    return new