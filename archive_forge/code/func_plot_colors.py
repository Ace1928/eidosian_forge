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
def plot_colors(self, xind, yind, **kws):
    """Plots color labels between the dendrogram and the heatmap

        Parameters
        ----------
        heatmap_kws : dict
            Keyword arguments heatmap

        """
    kws = kws.copy()
    kws.pop('cmap', None)
    kws.pop('norm', None)
    kws.pop('center', None)
    kws.pop('annot', None)
    kws.pop('vmin', None)
    kws.pop('vmax', None)
    kws.pop('robust', None)
    kws.pop('xticklabels', None)
    kws.pop('yticklabels', None)
    if self.row_colors is not None:
        matrix, cmap = self.color_list_to_matrix_and_cmap(self.row_colors, yind, axis=0)
        if self.row_color_labels is not None:
            row_color_labels = self.row_color_labels
        else:
            row_color_labels = False
        heatmap(matrix, cmap=cmap, cbar=False, ax=self.ax_row_colors, xticklabels=row_color_labels, yticklabels=False, **kws)
        if row_color_labels is not False:
            plt.setp(self.ax_row_colors.get_xticklabels(), rotation=90)
    else:
        despine(self.ax_row_colors, left=True, bottom=True)
    if self.col_colors is not None:
        matrix, cmap = self.color_list_to_matrix_and_cmap(self.col_colors, xind, axis=1)
        if self.col_color_labels is not None:
            col_color_labels = self.col_color_labels
        else:
            col_color_labels = False
        heatmap(matrix, cmap=cmap, cbar=False, ax=self.ax_col_colors, xticklabels=False, yticklabels=col_color_labels, **kws)
        if col_color_labels is not False:
            self.ax_col_colors.yaxis.tick_right()
            plt.setp(self.ax_col_colors.get_yticklabels(), rotation=0)
    else:
        despine(self.ax_col_colors, left=True, bottom=True)