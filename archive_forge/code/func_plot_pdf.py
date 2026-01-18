from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from statsmodels.graphics import utils
def plot_pdf(self, ticks_nbr=10, ax=None):
    """Plot the PDF.

        Parameters
        ----------
        ticks_nbr : int, optional
            Number of color isolines for the PDF. Default is 10.
        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.

        Returns
        -------
        fig : Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.

        """
    from matplotlib import pyplot as plt
    if self.k_dim != 2:
        import warnings
        warnings.warn('Plotting 2-dimensional Copula.')
    n_samples = 100
    eps = 0.0001
    uu, vv = np.meshgrid(np.linspace(eps, 1 - eps, n_samples), np.linspace(eps, 1 - eps, n_samples))
    points = np.vstack([uu.ravel(), vv.ravel()]).T
    data = self.pdf(points).T.reshape(uu.shape)
    min_ = np.nanpercentile(data, 5)
    max_ = np.nanpercentile(data, 95)
    fig, ax = utils.create_mpl_ax(ax)
    vticks = np.linspace(min_, max_, num=ticks_nbr)
    range_cbar = [min_, max_]
    cs = ax.contourf(uu, vv, data, vticks, antialiased=True, vmin=range_cbar[0], vmax=range_cbar[1])
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    cbar = plt.colorbar(cs, ticks=vticks)
    cbar.set_label('p')
    fig.tight_layout()
    return fig