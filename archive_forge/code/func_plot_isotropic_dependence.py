from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
def plot_isotropic_dependence(self, ax=None, xpoints=10, min_n=50):
    """
        Create a plot of the pairwise products of within-group
        residuals against the corresponding time differences.  This
        plot can be used to assess the possible form of an isotropic
        covariance structure.

        Parameters
        ----------
        ax : AxesSubplot
            An axes on which to draw the graph.  If None, new
            figure and axes objects are created
        xpoints : scalar or array_like
            If scalar, the number of points equally spaced points on
            the time difference axis used to define bins for
            calculating local means.  If an array, the specific points
            that define the bins.
        min_n : int
            The minimum sample size in a bin for the mean residual
            product to be included on the plot.
        """
    from statsmodels.graphics import utils as gutils
    resid = self.model.cluster_list(self.resid)
    time = self.model.cluster_list(self.model.time)
    xre, xdt = ([], [])
    for re, ti in zip(resid, time):
        ix = np.tril_indices(re.shape[0], 0)
        re = re[ix[0]] * re[ix[1]] / self.scale ** 2
        xre.append(re)
        dists = np.sqrt(((ti[ix[0], :] - ti[ix[1], :]) ** 2).sum(1))
        xdt.append(dists)
    xre = np.concatenate(xre)
    xdt = np.concatenate(xdt)
    if ax is None:
        fig, ax = gutils.create_mpl_ax(ax)
    else:
        fig = ax.get_figure()
    ii = np.flatnonzero(xdt == 0)
    v0 = np.mean(xre[ii])
    xre /= v0
    if np.isscalar(xpoints):
        xpoints = np.linspace(0, max(xdt), xpoints)
    dg = np.digitize(xdt, xpoints)
    dgu = np.unique(dg)
    hist = np.asarray([np.sum(dg == k) for k in dgu])
    ii = np.flatnonzero(hist >= min_n)
    dgu = dgu[ii]
    dgy = np.asarray([np.mean(xre[dg == k]) for k in dgu])
    dgx = np.asarray([np.mean(xdt[dg == k]) for k in dgu])
    ax.plot(dgx, dgy, '-', color='orange', lw=5)
    ax.set_xlabel('Time difference')
    ax.set_ylabel('Product of scaled residuals')
    return fig