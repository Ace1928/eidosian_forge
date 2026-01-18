from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def solve_and_plot_series(self, x0, params, varied_data, varied_idx, solver=None, plot_kwargs=None, plot_residuals_kwargs=None, **kwargs):
    """ Solve and plot for a series of a varied parameter.

        Convenience method, see :meth:`solve_series`, :meth:`plot_series` &
        :meth:`plot_series_residuals_internal` for more information.
        """
    sol, nfo = self.solve_series(x0, params, varied_data, varied_idx, solver=solver, **kwargs)
    ax_sol = self.plot_series(sol, varied_data, varied_idx, info=nfo, **plot_kwargs or {})
    extra = dict(ax_sol=ax_sol, info=nfo)
    if plot_residuals_kwargs:
        extra['ax_resid'] = self.plot_series_residuals_internal(varied_data, varied_idx, info=nfo, **plot_residuals_kwargs or {})
    return (sol, extra)