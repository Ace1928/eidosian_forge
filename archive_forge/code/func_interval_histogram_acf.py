from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
def interval_histogram_acf(activations, min_tau=1, max_tau=None):
    """
    Compute the interval histogram of the given (beat) activation function via
    auto-correlation as in [1]_.

    Parameters
    ----------
    activations : numpy array
        Beat activation function.
    min_tau : int, optional
        Minimal delay for the auto-correlation function [frames].
    max_tau : int, optional
        Maximal delay for the auto-correlation function [frames].

    Returns
    -------
    histogram_bins : numpy array
        Bins of the tempo histogram.
    histogram_delays : numpy array
        Corresponding delays [frames].

    References
    ----------
    .. [1] Sebastian Böck and Markus Schedl,
           "Enhanced Beat Tracking with Context-Aware Neural Networks",
           Proceedings of the 14th International Conference on Digital Audio
           Effects (DAFx), 2011.

    """
    if activations.ndim != 1:
        raise NotImplementedError('too many dimensions for autocorrelation interval histogram calculation.')
    if max_tau is None:
        max_tau = len(activations) - min_tau
    taus = list(range(min_tau, max_tau + 1))
    bins = []
    for tau in taus:
        bins.append(np.sum(np.abs(activations[tau:] * activations[0:-tau])))
    return (np.array(bins), np.array(taus))