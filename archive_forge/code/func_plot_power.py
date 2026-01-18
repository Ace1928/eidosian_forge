import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def plot_power(self, dep_var='nobs', nobs=None, effect_size=None, alpha=0.05, ax=None, title=None, plt_kwds=None, **kwds):
    """
        Plot power with number of observations or effect size on x-axis

        Parameters
        ----------
        dep_var : {'nobs', 'effect_size', 'alpha'}
            This specifies which variable is used for the horizontal axis.
            If dep_var='nobs' (default), then one curve is created for each
            value of ``effect_size``. If dep_var='effect_size' or alpha, then
            one curve is created for each value of ``nobs``.
        nobs : {scalar, array_like}
            specifies the values of the number of observations in the plot
        effect_size : {scalar, array_like}
            specifies the values of the effect_size in the plot
        alpha : {float, array_like}
            The significance level (type I error) used in the power
            calculation. Can only be more than a scalar, if ``dep_var='alpha'``
        ax : None or axis instance
            If ax is None, than a matplotlib figure is created. If ax is a
            matplotlib axis instance, then it is reused, and the plot elements
            are created with it.
        title : str
            title for the axis. Use an empty string, ``''``, to avoid a title.
        plt_kwds : {None, dict}
            not used yet
        kwds : dict
            These remaining keyword arguments are used as arguments to the
            power function. Many power function support ``alternative`` as a
            keyword argument, two-sample test support ``ratio``.

        Returns
        -------
        Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.

        Notes
        -----
        This works only for classes where the ``power`` method has
        ``effect_size``, ``nobs`` and ``alpha`` as the first three arguments.
        If the second argument is ``nobs1``, then the number of observations
        in the plot are those for the first sample.
        TODO: fix this for FTestPower and GofChisquarePower

        TODO: maybe add line variable, if we want more than nobs and effectsize
        """
    from statsmodels.graphics import utils
    from statsmodels.graphics.plottools import rainbow
    fig, ax = utils.create_mpl_ax(ax)
    import matplotlib.pyplot as plt
    colormap = plt.cm.Dark2
    plt_alpha = 1
    lw = 2
    if dep_var == 'nobs':
        colors = rainbow(len(effect_size))
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(effect_size))]
        for ii, es in enumerate(effect_size):
            power = self.power(es, nobs, alpha, **kwds)
            ax.plot(nobs, power, lw=lw, alpha=plt_alpha, color=colors[ii], label='es=%4.2F' % es)
            xlabel = 'Number of Observations'
    elif dep_var in ['effect size', 'effect_size', 'es']:
        colors = rainbow(len(nobs))
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(nobs))]
        for ii, n in enumerate(nobs):
            power = self.power(effect_size, n, alpha, **kwds)
            ax.plot(effect_size, power, lw=lw, alpha=plt_alpha, color=colors[ii], label='N=%4.2F' % n)
            xlabel = 'Effect Size'
    elif dep_var in ['alpha']:
        colors = rainbow(len(nobs))
        for ii, n in enumerate(nobs):
            power = self.power(effect_size, n, alpha, **kwds)
            ax.plot(alpha, power, lw=lw, alpha=plt_alpha, color=colors[ii], label='N=%4.2F' % n)
            xlabel = 'alpha'
    else:
        raise ValueError('depvar not implemented')
    if title is None:
        title = 'Power of Test'
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc='lower right')
    return fig