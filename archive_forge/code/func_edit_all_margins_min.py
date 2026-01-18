import itertools
import kiwisolver as kiwi
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
def edit_all_margins_min(self, todo, size):
    """
        Change the minimum size of all the margin of all
        the cells in the layout grid.

        Parameters
        ----------
        todo : {'left', 'right', 'bottom', 'top'}
            The margin to alter.

        size : float
            Minimum size of the margin.  If it is larger than the
            existing minimum it updates the margin size. Fraction of
            figure size.
        """
    for i in range(len(self.margin_vals[todo])):
        self.edit_margin_min(todo, size, i)