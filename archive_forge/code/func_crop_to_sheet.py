import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def crop_to_sheet(self, sheet_coord_system):
    """Crop the slice to the SheetCoordinateSystem's bounds."""
    maxrow, maxcol = sheet_coord_system.shape
    self[0] = max(0, self[0])
    self[1] = min(maxrow, self[1])
    self[2] = max(0, self[2])
    self[3] = min(maxcol, self[3])