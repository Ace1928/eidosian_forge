from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.tsa.vector_ar.util as util
def irf_grid_plot(values, stderr, impcol, rescol, names, title, signif=0.05, hlines=None, subplot_params=None, plot_params=None, figsize=(10, 10), stderr_type='asym'):
    """
    Reusable function to make flexible grid plots of impulse responses and
    comulative effects

    values : (T + 1) x k x k
    stderr : T x k x k
    hlines : k x k
    """
    import matplotlib.pyplot as plt
    if subplot_params is None:
        subplot_params = {}
    if plot_params is None:
        plot_params = {}
    nrows, ncols, to_plot = _get_irf_plot_config(names, impcol, rescol)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, squeeze=False, figsize=figsize)
    adjust_subplots()
    fig.suptitle(title, fontsize=14)
    subtitle_temp = '%s$\\rightarrow$%s'
    k = len(names)
    rng = lrange(len(values))
    for j, i, ai, aj in to_plot:
        ax = axes[ai][aj]
        if stderr is not None:
            if stderr_type == 'asym':
                sig = np.sqrt(stderr[:, j * k + i, j * k + i])
                plot_with_error(values[:, i, j], sig, x=rng, axes=ax, alpha=signif, value_fmt='b', stderr_type=stderr_type)
            if stderr_type in ('mc', 'sz1', 'sz2', 'sz3'):
                errs = (stderr[0][:, i, j], stderr[1][:, i, j])
                plot_with_error(values[:, i, j], errs, x=rng, axes=ax, alpha=signif, value_fmt='b', stderr_type=stderr_type)
        else:
            plot_with_error(values[:, i, j], None, x=rng, axes=ax, value_fmt='b')
        ax.axhline(0, color='k')
        if hlines is not None:
            ax.axhline(hlines[i, j], color='k')
        sz = subplot_params.get('fontsize', 12)
        ax.set_title(subtitle_temp % (names[j], names[i]), fontsize=sz)
    return fig