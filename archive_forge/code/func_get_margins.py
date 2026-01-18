import itertools
import kiwisolver as kiwi
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
def get_margins(self, todo, col):
    """Return the margin at this position"""
    return self.margin_vals[todo][col]