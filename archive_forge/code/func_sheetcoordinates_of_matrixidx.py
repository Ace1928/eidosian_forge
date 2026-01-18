import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def sheetcoordinates_of_matrixidx(self):
    """
        Return x,y where x is a vector of sheet coordinates
        representing the x-center of each matrix cell, and y
        represents the corresponding y-center of the cell.
        """
    rows, cols = self.shape
    return self.matrixidx2sheet(np.arange(rows), np.arange(cols))