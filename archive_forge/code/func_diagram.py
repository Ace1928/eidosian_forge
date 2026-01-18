import fractions
import functools
import re
from collections import OrderedDict
from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial import ConvexHull
import ase.units as units
from ase.formula import Formula
def diagram(self, U, pH, plot=True, show=False, ax=None):
    """Calculate Pourbaix diagram.

        U: list of float
            Potentials in V.
        pH: list of float
            pH values.
        plot: bool
            Create plot.
        show: bool
            Open graphical window and show plot.
        ax: matplotlib axes object
            When creating plot, plot onto the given axes object.
            If none given, plot onto the current one.
        """
    a = np.empty((len(U), len(pH)), int)
    a[:] = -1
    colors = {}
    f = functools.partial(self.colorfunction, colors=colors)
    bisect(a, U, pH, f)
    compositions = [None] * len(colors)
    names = [ref[-1] for ref in self.references]
    for indices, color in colors.items():
        compositions[color] = ' + '.join((names[i] for i in indices if names[i] not in ['H2O(aq)', 'H+(aq)', 'e-']))
    text = []
    for i, name in enumerate(compositions):
        b = a == i
        x = np.dot(b.sum(1), U) / b.sum()
        y = np.dot(b.sum(0), pH) / b.sum()
        name = re.sub('(\\S)([+-]+)', '\\1$^{\\2}$', name)
        name = re.sub('(\\d+)', '$_{\\1}$', name)
        text.append((x, y, name))
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        if ax is None:
            ax = plt.gca()
        ax.imshow(a, cmap=cm.Accent, extent=[min(pH), max(pH), min(U), max(U)], origin='lower', aspect='auto')
        for x, y, name in text:
            ax.text(y, x, name, horizontalalignment='center')
        ax.set_xlabel('pH')
        ax.set_ylabel('potential [V]')
        ax.set_xlim(min(pH), max(pH))
        ax.set_ylim(min(U), max(U))
        if show:
            plt.show()
    return (a, compositions, text)