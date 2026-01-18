from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.tsa.vector_ar.util as util
def plot_full_acorr(acorr, fontsize=8, linewidth=8, xlabel=None, err_bound=None):
    """

    Parameters
    ----------
    """
    import matplotlib.pyplot as plt
    config = MPLConfigurator()
    config.set_fontsize(fontsize)
    k = acorr.shape[1]
    fig, axes = plt.subplots(k, k, figsize=(10, 10), squeeze=False)
    for i in range(k):
        for j in range(k):
            ax = axes[i][j]
            acorr_plot(acorr[:, i, j], linewidth=linewidth, xlabel=xlabel, ax=ax)
            if err_bound is not None:
                ax.axhline(err_bound, color='k', linestyle='--')
                ax.axhline(-err_bound, color='k', linestyle='--')
    adjust_subplots()
    config.revert()
    return fig