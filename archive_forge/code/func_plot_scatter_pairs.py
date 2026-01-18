import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def plot_scatter_pairs(self, idx1, idx2, title=None, ax=None):
    """create scatter plot of two random effects

        Parameters
        ----------
        idx1, idx2 : int
            indices of the two random effects to display, corresponding to
            columns of exog_re
        title : None or string
            If None, then a default title is added
        ax : None or matplotlib axis instance
            If None, then a figure with one axis is created and returned.
            If ax is not None, then the scatter plot is created on it, and
            this axis instance is returned.

        Returns
        -------
        ax_or_fig : axis or figure instance
            see ax parameter

        Notes
        -----
        Still needs ellipse from estimated parameters

        """
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax_or_fig = fig
    re1 = self.params_random_units[:, idx1]
    re2 = self.params_random_units[:, idx2]
    ax.plot(re1, re2, 'o', alpha=0.75)
    if title is None:
        title = 'Random Effects %d and %d' % (idx1, idx2)
    ax.set_title(title)
    ax_or_fig = ax
    return ax_or_fig