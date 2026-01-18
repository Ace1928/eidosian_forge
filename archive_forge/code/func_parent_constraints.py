import itertools
import kiwisolver as kiwi
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
def parent_constraints(self, parent):
    if not isinstance(parent, LayoutGrid):
        hc = [self.lefts[0] == parent[0], self.rights[-1] == parent[0] + parent[2], self.tops[0] == parent[1] + parent[3], self.bottoms[-1] == parent[1]]
    else:
        rows, cols = self.parent_pos
        rows = np.atleast_1d(rows)
        cols = np.atleast_1d(cols)
        left = parent.lefts[cols[0]]
        right = parent.rights[cols[-1]]
        top = parent.tops[rows[0]]
        bottom = parent.bottoms[rows[-1]]
        if self.parent_inner:
            left += parent.margins['left'][cols[0]]
            left += parent.margins['leftcb'][cols[0]]
            right -= parent.margins['right'][cols[-1]]
            right -= parent.margins['rightcb'][cols[-1]]
            top -= parent.margins['top'][rows[0]]
            top -= parent.margins['topcb'][rows[0]]
            bottom += parent.margins['bottom'][rows[-1]]
            bottom += parent.margins['bottomcb'][rows[-1]]
        hc = [self.lefts[0] == left, self.rights[-1] == right, self.tops[0] == top, self.bottoms[-1] == bottom]
    for c in hc:
        self.solver.addConstraint(c | 'required')