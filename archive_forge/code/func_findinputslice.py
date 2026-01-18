import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
@staticmethod
def findinputslice(coord, sliceshape, sheetshape):
    """
        Gets the matrix indices of a slice within an array of size
        sheetshape from a sliceshape, positioned at coord.
        """
    center_row, center_col = coord
    n_rows, n_cols = sliceshape
    sheet_rows, sheet_cols = sheetshape
    c1 = -min(0, center_col - n_cols / 2)
    r1 = -min(0, center_row - n_rows / 2)
    c2 = -max(-n_cols, center_col - sheet_cols - n_cols / 2)
    r2 = -max(-n_rows, center_row - sheet_rows - n_rows / 2)
    return (r1, r2, c1, c2)