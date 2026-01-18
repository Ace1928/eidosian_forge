from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def label_cells(self, func):
    """Return None.  Labels cells based on `func`.
        If ``func(cell) is None`` then its datatype is
        not changed; otherwise it is set to ``func(cell)``.
        """
    for row in self:
        for cell in row:
            label = func(cell)
            if label is not None:
                cell.datatype = label