import pickle
import sys
import numpy as np
from ase.gui.i18n import _
import ase.gui.ui as ui
def make_plot(data, i, expr, type, show=True):
    import matplotlib.pyplot as plt
    basesize = 4
    plt.figure(figsize=(basesize * 2.5 ** 0.5, basesize))
    m = len(data)
    if type is None:
        if m == 1:
            type = 'y'
        else:
            type = 'xy'
    if type == 'y':
        for j in range(m):
            plt.plot(data[j])
            plt.plot([i], [data[j, i]], 'o')
    else:
        for j in range(1, m):
            plt.plot(data[0], data[j])
            plt.plot([data[0, i]], [data[j, i]], 'o')
    plt.title(expr)
    if show:
        plt.show()