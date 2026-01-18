import fractions
import functools
import re
from collections import OrderedDict
from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial import ConvexHull
import ase.units as units
from ase.formula import Formula
def plot2d3(self, ax=None):
    x, y = self.points[:, 1:-1].T.copy()
    x += y / 2
    y *= 3 ** 0.5 / 2
    names = [re.sub('(\\d+)', '$_{\\1}$', ref[2]) for ref in self.references]
    hull = self.hull
    simplices = self.simplices
    if ax:
        for i, j, k in simplices:
            ax.plot(x[[i, j, k, i]], y[[i, j, k, i]], '-b')
        ax.plot(x[hull], y[hull], 'og')
        ax.plot(x[~hull], y[~hull], 'sr')
        for a, b, name in zip(x, y, names):
            ax.text(a, b, name, ha='center', va='top')
    return (x, y, names, hull, simplices)