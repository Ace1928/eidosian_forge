import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.base import HolderTuple
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS
def plot_probs(freq, probs_predicted, label='predicted', upp_xlim=None, fig=None):
    """diagnostic plots for comparing two lists of discrete probabilities

    Parameters
    ----------
    freq, probs_predicted : nd_arrays
        two arrays of probabilities, this can be any probabilities for
        the same events, default is designed for comparing predicted
        and observed probabilities
    label : str or tuple
        If string, then it will be used as the label for probs_predicted and
        "freq" is used for the other probabilities.
        If label is a tuple of strings, then the first is they are used as
        label for both probabilities

    upp_xlim : None or int
        If it is not None, then the xlim of the first two plots are set to
        (0, upp_xlim), otherwise the matplotlib default is used
    fig : None or matplotlib figure instance
        If fig is provided, then the axes will be added to it in a (3,1)
        subplots, otherwise a matplotlib figure instance is created

    Returns
    -------
    Figure
        The figure contains 3 subplot with probabilities, cumulative
        probabilities and a PP-plot
    """
    if isinstance(label, list):
        label0, label1 = label
    else:
        label0, label1 = ('freq', label)
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 12))
    ax1 = fig.add_subplot(311)
    ax1.plot(freq, '-o', label=label0)
    ax1.plot(probs_predicted, '-d', label=label1)
    if upp_xlim is not None:
        ax1.set_xlim(0, upp_xlim)
    ax1.legend()
    ax1.set_title('probabilities')
    ax2 = fig.add_subplot(312)
    ax2.plot(np.cumsum(freq), '-o', label=label0)
    ax2.plot(np.cumsum(probs_predicted), '-d', label=label1)
    if upp_xlim is not None:
        ax2.set_xlim(0, upp_xlim)
    ax2.legend()
    ax2.set_title('cumulative probabilities')
    ax3 = fig.add_subplot(313)
    ax3.plot(np.cumsum(probs_predicted), np.cumsum(freq), 'o')
    ax3.plot(np.arange(len(freq)) / len(freq), np.arange(len(freq)) / len(freq))
    ax3.set_title('PP-plot')
    ax3.set_xlabel(label1)
    ax3.set_ylabel(label0)
    return fig