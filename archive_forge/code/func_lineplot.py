import copy
from textwrap import dedent
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import utils
from . import algorithms as algo
from .axisgrid import FacetGrid, _facet_docs
def lineplot(self, ax, kws):
    """Draw the model."""
    grid, yhat, err_bands = self.fit_regression(ax)
    edges = (grid[0], grid[-1])
    fill_color = kws['color']
    lw = kws.pop('lw', mpl.rcParams['lines.linewidth'] * 1.5)
    kws.setdefault('linewidth', lw)
    line, = ax.plot(grid, yhat, **kws)
    if not self.truncate:
        line.sticky_edges.x[:] = edges
    if err_bands is not None:
        ax.fill_between(grid, *err_bands, facecolor=fill_color, alpha=0.15)