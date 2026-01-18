from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.graphics import utils
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import GLS, OLS, WLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tools.tools import maybe_unwrap_results
from ._regressionplots_doc import (
def plot_partregress_grid(results, exog_idx=None, grid=None, fig=None):
    """
    Plot partial regression for a set of regressors.

    Parameters
    ----------
    results : Results instance
        A regression model results instance.
    exog_idx : {None, list[int], list[str]}
        The indices  or column names of the exog used in the plot, default is
        all.
    grid : {None, tuple[int]}
        If grid is given, then it is used for the arrangement of the subplots.
        The format of grid is  (nrows, ncols). If grid is None, then ncol is
        one, if there are only 2 subplots, and the number of columns is two
        otherwise.
    fig : Figure, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    Figure
        If `fig` is None, the created figure.  Otherwise `fig` itself.

    See Also
    --------
    plot_partregress : Plot partial regression for a single regressor.
    plot_ccpr : Plot CCPR against one regressor

    Notes
    -----
    A subplot is created for each explanatory variable given by exog_idx.
    The partial regression plot shows the relationship between the response
    and the given explanatory variable after removing the effect of all other
    explanatory variables in exog.

    References
    ----------
    See http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/partregr.htm

    Examples
    --------
    Using the state crime dataset separately plot the effect of the each
    variable on the on the outcome, murder rate while accounting for the effect
    of all other variables in the model visualized with a grid of partial
    regression plots.

    >>> from statsmodels.graphics.regressionplots import plot_partregress_grid
    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.formula.api as smf

    >>> fig = plt.figure(figsize=(8, 6))
    >>> crime_data = sm.datasets.statecrime.load_pandas()
    >>> results = smf.ols('murder ~ hs_grad + urban + poverty + single',
    ...                   data=crime_data.data).fit()
    >>> plot_partregress_grid(results, fig=fig)
    >>> plt.show()

    .. plot:: plots/graphics_regression_partregress_grid.py
    """
    import pandas
    fig = utils.create_mpl_fig(fig)
    exog_name, exog_idx = utils.maybe_name_or_idx(exog_idx, results.model)
    y = pandas.Series(results.model.endog, name=results.model.endog_names)
    exog = results.model.exog
    k_vars = exog.shape[1]
    nrows = (len(exog_idx) + 1) // 2
    ncols = 1 if nrows == len(exog_idx) else 2
    if grid is not None:
        nrows, ncols = grid
    if ncols > 1:
        title_kwargs = {'fontdict': {'fontsize': 'small'}}
    other_names = np.array(results.model.exog_names)
    for i, idx in enumerate(exog_idx):
        others = lrange(k_vars)
        others.pop(idx)
        exog_others = pandas.DataFrame(exog[:, others], columns=other_names[others])
        ax = fig.add_subplot(nrows, ncols, i + 1)
        plot_partregress(y, pandas.Series(exog[:, idx], name=other_names[idx]), exog_others, ax=ax, title_kwargs=title_kwargs, obs_labels=False)
        ax.set_title('')
    fig.suptitle('Partial Regression Plot', fontsize='large')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    return fig