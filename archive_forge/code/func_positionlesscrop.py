import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def positionlesscrop(self, x, y, sheet_coord_system):
    """
        Return the correct slice for a weights/mask matrix at this
        ConnectionField's location on the sheet (i.e. for getting the
        correct submatrix of the weights or mask in case the unit is
        near the edge of the sheet).
        """
    slice_inds = self.findinputslice(sheet_coord_system.sheet2matrixidx(x, y), self.shape_on_sheet(), sheet_coord_system.shape)
    self.set(slice_inds)