import fractions
import functools
import re
from collections import OrderedDict
from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial import ConvexHull
import ase.units as units
from ase.formula import Formula
def plot2d2(self, ax=None, only_label_simplices=False, only_plot_simplices=False):
    x, e = self.points[:, 1:].T
    names = [re.sub('(\\d+)', '$_{\\1}$', ref[2]) for ref in self.references]
    hull = self.hull
    simplices = self.simplices
    xlabel = self.symbols[1]
    ylabel = 'energy [eV/atom]'
    if ax:
        for i, j in simplices:
            ax.plot(x[[i, j]], e[[i, j]], '-b')
        ax.plot(x[hull], e[hull], 'sg')
        if not only_plot_simplices:
            ax.plot(x[~hull], e[~hull], 'or')
        if only_plot_simplices or only_label_simplices:
            x = x[self.hull]
            e = e[self.hull]
            names = [name for name, h in zip(names, self.hull) if h]
        for a, b, name in zip(x, e, names):
            ax.text(a, b, name, ha='center', va='top')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    return (x, e, names, hull, simplices, xlabel, ylabel)