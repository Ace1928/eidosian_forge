import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
@staticmethod
def color_list_to_matrix_and_cmap(colors, ind, axis=0):
    """Turns a list of colors into a numpy matrix and matplotlib colormap

        These arguments can now be plotted using heatmap(matrix, cmap)
        and the provided colors will be plotted.

        Parameters
        ----------
        colors : list of matplotlib colors
            Colors to label the rows or columns of a dataframe.
        ind : list of ints
            Ordering of the rows or columns, to reorder the original colors
            by the clustered dendrogram order
        axis : int
            Which axis this is labeling

        Returns
        -------
        matrix : numpy.array
            A numpy array of integer values, where each indexes into the cmap
        cmap : matplotlib.colors.ListedColormap

        """
    try:
        mpl.colors.to_rgb(colors[0])
    except ValueError:
        m, n = (len(colors), len(colors[0]))
        if not all((len(c) == n for c in colors[1:])):
            raise ValueError('Multiple side color vectors must have same size')
    else:
        m, n = (1, len(colors))
        colors = [colors]
    unique_colors = {}
    matrix = np.zeros((m, n), int)
    for i, inner in enumerate(colors):
        for j, color in enumerate(inner):
            idx = unique_colors.setdefault(color, len(unique_colors))
            matrix[i, j] = idx
    matrix = matrix[:, ind]
    if axis == 0:
        matrix = matrix.T
    cmap = mpl.colors.ListedColormap(list(unique_colors))
    return (matrix, cmap)