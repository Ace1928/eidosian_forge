from warnings import warn
import numpy as np
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import OutputWarning, SpecificationWarning
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.tsatools import lagmat
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .initialization import Initialization
from .tools import (
def plot_components(self, which=None, alpha=0.05, observed=True, level=True, trend=True, seasonal=True, freq_seasonal=True, cycle=True, autoregressive=True, legend_loc='upper right', fig=None, figsize=None):
    """
        Plot the estimated components of the model.

        Parameters
        ----------
        which : {'filtered', 'smoothed'}, or None, optional
            Type of state estimate to plot. Default is 'smoothed' if smoothed
            results are available otherwise 'filtered'.
        alpha : float, optional
            The confidence intervals for the components are (1 - alpha) %
        observed : bool, optional
            Whether or not to plot the observed series against
            one-step-ahead predictions.
            Default is True.
        level : bool, optional
            Whether or not to plot the level component, if applicable.
            Default is True.
        trend : bool, optional
            Whether or not to plot the trend component, if applicable.
            Default is True.
        seasonal : bool, optional
            Whether or not to plot the seasonal component, if applicable.
            Default is True.
        freq_seasonal : bool, optional
            Whether or not to plot the frequency domain seasonal component(s),
            if applicable. Default is True.
        cycle : bool, optional
            Whether or not to plot the cyclical component, if applicable.
            Default is True.
        autoregressive : bool, optional
            Whether or not to plot the autoregressive state, if applicable.
            Default is True.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        If all options are included in the model and selected, this produces
        a 6x1 plot grid with the following plots (ordered top-to-bottom):

        0. Observed series against predicted series
        1. Level
        2. Trend
        3. Seasonal
        4. Freq Seasonal
        5. Cycle
        6. Autoregressive

        Specific subplots will be removed if the component is not present in
        the estimated model or if the corresponding keyword argument is set to
        False.

        All plots contain (1 - `alpha`) %  confidence intervals.
        """
    from scipy.stats import norm
    from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
    plt = _import_mpl()
    fig = create_mpl_fig(fig, figsize)
    if which is None:
        which = 'filtered' if self.smoothed_state is None else 'smoothed'
    spec = self.specification
    comp = [('level', level and spec.level), ('trend', trend and spec.trend), ('seasonal', seasonal and spec.seasonal)]
    if freq_seasonal and spec.freq_seasonal:
        for ix, _ in enumerate(spec.freq_seasonal_periods):
            key = f'freq_seasonal_{ix!r}'
            comp.append((key, True))
    comp.extend([('cycle', cycle and spec.cycle), ('autoregressive', autoregressive and spec.autoregressive)])
    components = dict(comp)
    llb = self.filter_results.loglikelihood_burn
    k_plots = observed + np.sum(list(components.values()))
    if hasattr(self.data, 'dates') and self.data.dates is not None:
        dates = self.data.dates._mpl_repr()
    else:
        dates = np.arange(len(self.data.endog))
    critical_value = norm.ppf(1 - alpha / 2.0)
    plot_idx = 1
    if observed:
        ax = fig.add_subplot(k_plots, 1, plot_idx)
        plot_idx += 1
        ax.plot(dates[llb:], self.model.endog[llb:], color='k', label='Observed')
        predict = self.filter_results.forecasts[0]
        std_errors = np.sqrt(self.filter_results.forecasts_error_cov[0, 0])
        ci_lower = predict - critical_value * std_errors
        ci_upper = predict + critical_value * std_errors
        ax.plot(dates[llb:], predict[llb:], label='One-step-ahead predictions')
        ci_poly = ax.fill_between(dates[llb:], ci_lower[llb:], ci_upper[llb:], alpha=0.2)
        ci_label = '$%.3g \\%%$ confidence interval' % ((1 - alpha) * 100)
        p = plt.Rectangle((0, 0), 1, 1, fc=ci_poly.get_facecolor()[0])
        handles, labels = ax.get_legend_handles_labels()
        handles.append(p)
        labels.append(ci_label)
        ax.legend(handles, labels, loc=legend_loc)
        ax.set_title('Predicted vs observed')
    for component, is_plotted in components.items():
        if not is_plotted:
            continue
        ax = fig.add_subplot(k_plots, 1, plot_idx)
        plot_idx += 1
        try:
            component_bunch = getattr(self, component)
            title = component.title()
        except AttributeError:
            if component.startswith('freq_seasonal_'):
                ix = int(component.replace('freq_seasonal_', ''))
                big_bunch = getattr(self, 'freq_seasonal')
                component_bunch = big_bunch[ix]
                title = component_bunch.pretty_name
            else:
                raise
        if which not in component_bunch:
            raise ValueError('Invalid type of state estimate.')
        which_cov = '%s_cov' % which
        value = component_bunch[which]
        state_label = '{} ({})'.format(title, which)
        ax.plot(dates[llb:], value[llb:], label=state_label)
        if which_cov in component_bunch:
            std_errors = np.sqrt(component_bunch['%s_cov' % which])
            ci_lower = value - critical_value * std_errors
            ci_upper = value + critical_value * std_errors
            ci_poly = ax.fill_between(dates[llb:], ci_lower[llb:], ci_upper[llb:], alpha=0.2)
            ci_label = '$%.3g \\%%$ confidence interval' % ((1 - alpha) * 100)
        ax.legend(loc=legend_loc)
        ax.set_title('%s component' % title)
    if llb > 0:
        text = 'Note: The first %d observations are not shown, due to approximate diffuse initialization.'
        fig.text(0.1, 0.01, text % llb, fontsize='large')
    return fig