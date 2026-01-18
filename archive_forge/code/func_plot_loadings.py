import warnings
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl
from .factor_rotation import rotate_factors, promax
def plot_loadings(self, loading_pairs=None, plot_prerotated=False):
    """
        Plot factor loadings in 2-d plots

        Parameters
        ----------
        loading_pairs : None or a list of tuples
            Specify plots. Each tuple (i, j) represent one figure, i and j is
            the loading number for x-axis and y-axis, respectively. If `None`,
            all combinations of the loadings will be plotted.
        plot_prerotated : True or False
            If True, the loadings before rotation applied will be plotted. If
            False, rotated loadings will be plotted.

        Returns
        -------
        figs : a list of figure handles
        """
    _import_mpl()
    from .plots import plot_loadings
    if self.rotation_method is None:
        plot_prerotated = True
    loadings = self.loadings_no_rot if plot_prerotated else self.loadings
    if plot_prerotated:
        title = 'Prerotated Factor Pattern'
    else:
        title = '%s Rotated Factor Pattern' % self.rotation_method
    var_explained = self.eigenvals / self.n_comp * 100
    return plot_loadings(loadings, loading_pairs=loading_pairs, title=title, row_names=self.endog_names, percent_variance=var_explained)