import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def plot_random_univariate(self, bins=None, use_loc=True):
    """create plot of marginal distribution of random effects

        Parameters
        ----------
        bins : int or bin edges
            option for bins in matplotlibs hist method. Current default is not
            very sophisticated. All distributions use the same setting for
            bins.
        use_loc : bool
            If True, then the distribution with mean given by the fixed
            effect is used.

        Returns
        -------
        Figure
            figure with subplots

        Notes
        -----
        What can make this fancier?

        Bin edges will not make sense if loc or scale differ across random
        effect distributions.

        """
    import matplotlib.pyplot as plt
    from scipy.stats import norm as normal
    fig = plt.figure()
    k = self.model.k_exog_re
    if k > 3:
        rows, cols = (int(np.ceil(k * 0.5)), 2)
    else:
        rows, cols = (k, 1)
    if bins is None:
        bins = 5 + 2 * self.model.n_units ** (1.0 / 3.0)
    if use_loc:
        loc = self.mean_random()
    else:
        loc = [0] * k
    scale = self.std_random()
    for ii in range(k):
        ax = fig.add_subplot(rows, cols, ii)
        freq, bins_, _ = ax.hist(loc[ii] + self.params_random_units[:, ii], bins=bins, normed=True)
        points = np.linspace(bins_[0], bins_[-1], 200)
        ax.set_title('Random Effect %d Marginal Distribution' % ii)
        ax.plot(points, normal.pdf(points, loc=loc[ii], scale=scale[ii]), 'r')
    return fig