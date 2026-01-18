import itertools
import kiwisolver as kiwi
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
def hard_constraints(self):
    """
        These are the redundant constraints, plus ones that make the
        rest of the code easier.
        """
    for i in range(self.ncols):
        hc = [self.rights[i] >= self.lefts[i], self.rights[i] - self.margins['right'][i] - self.margins['rightcb'][i] >= self.lefts[i] - self.margins['left'][i] - self.margins['leftcb'][i]]
        for c in hc:
            self.solver.addConstraint(c | 'required')
    for i in range(self.nrows):
        hc = [self.tops[i] >= self.bottoms[i], self.tops[i] - self.margins['top'][i] - self.margins['topcb'][i] >= self.bottoms[i] - self.margins['bottom'][i] - self.margins['bottomcb'][i]]
        for c in hc:
            self.solver.addConstraint(c | 'required')