import itertools
import kiwisolver as kiwi
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
def edit_margins(self, todo, size):
    """
        Change the size of all the margin of all the cells in the layout grid.

        Parameters
        ----------
        todo : string (one of 'left', 'right', 'bottom', 'top')
            margin to alter.

        size : float
            Size to set the margins.  Fraction of figure size.
        """
    for i in range(len(self.margin_vals[todo])):
        self.edit_margin(todo, size, i)