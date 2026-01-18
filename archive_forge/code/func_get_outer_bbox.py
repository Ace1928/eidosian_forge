import itertools
import kiwisolver as kiwi
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
def get_outer_bbox(self, rows=0, cols=0):
    """
        Return the outer bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
    rows = np.atleast_1d(rows)
    cols = np.atleast_1d(cols)
    bbox = Bbox.from_extents(self.lefts[cols[0]].value(), self.bottoms[rows[-1]].value(), self.rights[cols[-1]].value(), self.tops[rows[0]].value())
    return bbox