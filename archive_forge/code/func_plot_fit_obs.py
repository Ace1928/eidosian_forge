import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def plot_fit_obs(self, col_name, lowess_args=None, lowess_min_n=40, jitter=None, plot_points=True, ax=None):
    """
        Plot fitted versus imputed or observed values as a scatterplot.

        Parameters
        ----------
        col_name : str
            The variable to be plotted on the horizontal axis.
        lowess_args : dict-like
            Keyword arguments passed to lowess fit.  A dictionary of
            dictionaries, keys are 'o' and 'i' denoting 'observed' and
            'imputed', respectively.
        lowess_min_n : int
            Minimum sample size to plot a lowess fit
        jitter : float or tuple
            Standard deviation for jittering points in the plot.
            Either a single scalar applied to both axes, or a tuple
            containing x-axis jitter and y-axis jitter, respectively.
        plot_points : bool
            If True, the data points are plotted.
        ax : AxesSubplot
            Axes on which to plot, created if not provided.

        Returns
        -------
        The matplotlib figure on which the plot is drawn.
        """
    from statsmodels.graphics import utils as gutils
    from statsmodels.nonparametric.smoothers_lowess import lowess
    if lowess_args is None:
        lowess_args = {}
    if ax is None:
        fig, ax = gutils.create_mpl_ax(ax)
    else:
        fig = ax.get_figure()
    ax.set_position([0.1, 0.1, 0.7, 0.8])
    ixi = self.ix_miss[col_name]
    ixo = self.ix_obs[col_name]
    vec1 = np.require(self.data[col_name], requirements='W')
    formula = self.conditional_formula[col_name]
    endog, exog = patsy.dmatrices(formula, self.data, return_type='dataframe')
    results = self.results[col_name]
    vec2 = results.predict(exog=exog)
    vec2 = self._get_predicted(vec2)
    if jitter is not None:
        if np.isscalar(jitter):
            jitter = (jitter, jitter)
        vec1 += jitter[0] * np.random.normal(size=len(vec1))
        vec2 += jitter[1] * np.random.normal(size=len(vec2))
    keys = ['o', 'i']
    ixs = {'o': ixo, 'i': ixi}
    lak = {'o': 'obs', 'i': 'imp'}
    color = {'o': 'orange', 'i': 'lime'}
    if plot_points:
        for ky in keys:
            ix = ixs[ky]
            ax.plot(vec1[ix], vec2[ix], 'o', color=color[ky], label=lak[ky], alpha=0.6)
    for ky in keys:
        ix = ixs[ky]
        if len(ix) < lowess_min_n:
            continue
        if ky in lowess_args:
            la = lowess_args[ky]
        else:
            la = {}
        ix = ixs[ky]
        lfit = lowess(vec2[ix], vec1[ix], **la)
        ax.plot(lfit[:, 0], lfit[:, 1], '-', color=color[ky], alpha=0.6, lw=4, label=lak[ky])
    ha, la = ax.get_legend_handles_labels()
    leg = fig.legend(ha, la, loc='center right', numpoints=1)
    leg.draw_frame(False)
    ax.set_xlabel(col_name + ' observed or imputed')
    ax.set_ylabel(col_name + ' fitted')
    return fig