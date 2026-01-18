import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def matrixidx2sheet(self, row, col):
    """
        Return (x,y) where x and y are the floating point coordinates
        of the *center* of the given matrix cell (row,col). If the
        matrix cell represents a 0.2 by 0.2 region, then the center
        location returned would be 0.1,0.1.

        NOTE: This is NOT the strict mathematical inverse of
        sheet2matrixidx(), because sheet2matrixidx() discards all but
        the integer portion of the continuous matrix coordinate.

        Valid only for scalar or array row and col.
        """
    x, y = self.matrix2sheet(row + 0.5, col + 0.5)
    if not isinstance(x, datetime_types):
        x = np.around(x, 10)
    if not isinstance(y, datetime_types):
        y = np.around(y, 10)
    return (x, y)