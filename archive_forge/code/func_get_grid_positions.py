import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
@_api.delete_parameter('3.7', 'raw')
def get_grid_positions(self, fig, raw=False):
    """
        Return the positions of the grid cells in figure coordinates.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure the grid should be applied to. The subplot parameters
            (margins and spacing between subplots) are taken from *fig*.
        raw : bool, default: False
            If *True*, the subplot parameters of the figure are not taken
            into account. The grid spans the range [0, 1] in both directions
            without margins and there is no space between grid cells. This is
            used for constrained_layout.

        Returns
        -------
        bottoms, tops, lefts, rights : array
            The bottom, top, left, right positions of the grid cells in
            figure coordinates.
        """
    nrows, ncols = self.get_geometry()
    if raw:
        left = 0.0
        right = 1.0
        bottom = 0.0
        top = 1.0
        wspace = 0.0
        hspace = 0.0
    else:
        subplot_params = self.get_subplot_params(fig)
        left = subplot_params.left
        right = subplot_params.right
        bottom = subplot_params.bottom
        top = subplot_params.top
        wspace = subplot_params.wspace
        hspace = subplot_params.hspace
    tot_width = right - left
    tot_height = top - bottom
    cell_h = tot_height / (nrows + hspace * (nrows - 1))
    sep_h = hspace * cell_h
    norm = cell_h * nrows / sum(self._row_height_ratios)
    cell_heights = [r * norm for r in self._row_height_ratios]
    sep_heights = [0] + [sep_h] * (nrows - 1)
    cell_hs = np.cumsum(np.column_stack([sep_heights, cell_heights]).flat)
    cell_w = tot_width / (ncols + wspace * (ncols - 1))
    sep_w = wspace * cell_w
    norm = cell_w * ncols / sum(self._col_width_ratios)
    cell_widths = [r * norm for r in self._col_width_ratios]
    sep_widths = [0] + [sep_w] * (ncols - 1)
    cell_ws = np.cumsum(np.column_stack([sep_widths, cell_widths]).flat)
    fig_tops, fig_bottoms = (top - cell_hs).reshape((-1, 2)).T
    fig_lefts, fig_rights = (left + cell_ws).reshape((-1, 2)).T
    return (fig_bottoms, fig_tops, fig_lefts, fig_rights)