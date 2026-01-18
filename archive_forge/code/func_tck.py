import operator
import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename
from cupyx.scipy.sparse import csr_matrix
import numpy as np
@property
def tck(self):
    """Equivalent to ``(self.t, self.c, self.k)`` (read-only).
        """
    return (self.t, self.c, self.k)