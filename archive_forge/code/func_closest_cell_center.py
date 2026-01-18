import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def closest_cell_center(self, x, y):
    """
        Given arbitrary sheet coordinates, return the sheet coordinates
        of the center of the closest unit.
        """
    return self.matrixidx2sheet(*self.sheet2matrixidx(x, y))