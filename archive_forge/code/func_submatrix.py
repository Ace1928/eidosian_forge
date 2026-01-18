import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def submatrix(self, matrix):
    """
        Return the submatrix of the given matrix specified by this
        slice.

        Equivalent to computing the intersection between the
        SheetCoordinateSystem's bounds and the bounds, and returning
        the corresponding submatrix of the given matrix.

        The submatrix is just a view into the sheet_matrix; it is not
        an independent copy.
        """
    return matrix[self[0]:self[1], self[2]:self[3]]