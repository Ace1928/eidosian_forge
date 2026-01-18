from ..units import latex_of_unit, is_unitless, to_unitless, unit_of
from ..printing import number_to_scientific_latex
def plot_avg_params(opt_params, cov_params, avg_params_result, label_cb=None, ax=None, title=False, xlabel=False, ylabel=False, flip=False, nsigma=1):
    """Calculates the average parameters from a set of regression parameters

    Parameters
    ----------
    opt_params : array_like
        Of shape ``(nfits, nparams)``.
    cov_params : array_like
        of shape (nfits, nparams, nparams)
    avg_params_result : length-2 tuple
       Result from :func:`avg_parrams`.
    label_cb : callable
        signature (beta, variance_beta) -> str
    ax : matplotlib.axes.Axes
    title : bool or str
    xlabel : bool or str
    ylabel : bool or str
    flip : bool
        for plotting: (x, y) -> beta1, beta0
    nsigma : int
        Multiplier for error bars

    Returns
    -------
    avg_beta: weighted average of parameters
    var_avg_beta: variance-covariance matrix

    """
    avg_beta, var_avg_beta = avg_params_result
    import matplotlib.pyplot as plt
    if label_cb is not None:
        lbl = label_cb(avg_beta, var_avg_beta)
    else:
        lbl = None
    if ax is None:
        ax = plt.subplot(1, 1, 1)
    xidx, yidx = (1, 0) if flip else (0, 1)
    opt_params = np.asarray(opt_params)
    cov_params = np.asarray(cov_params)
    var_beta = np.vstack((cov_params[:, 0, 0], cov_params[:, 1, 1])).T
    ax.errorbar(opt_params[:, xidx], opt_params[:, yidx], marker='s', ls='None', xerr=nsigma * var_beta[:, xidx] ** 0.5, yerr=nsigma * var_beta[:, yidx] ** 0.5)
    if xlabel:
        if xlabel is True:
            xlabel = '$\\beta_%d$' % xidx
        ax.set_xlabel(xlabel)
    if ylabel:
        if ylabel is True:
            xlabel = '$\\beta_%d$' % yidx
        ax.set_ylabel(ylabel)
    if title:
        if title is True:
            title = '$y(x) = \\beta_0 + \\beta_1 \\cdot x$'
        ax.set_title(title)
    ax.errorbar(avg_beta[xidx], avg_beta[yidx], xerr=nsigma * var_avg_beta[xidx] ** 0.5, yerr=nsigma * var_avg_beta[yidx] ** 0.5, marker='o', c='r', linewidth=2, markersize=10, label=lbl)
    ax.legend(numpoints=1)