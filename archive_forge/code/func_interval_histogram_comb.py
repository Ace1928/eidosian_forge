from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
def interval_histogram_comb(activations, alpha, min_tau=1, max_tau=None):
    """
    Compute the interval histogram of the given (beat) activation function via
    a bank of resonating comb filters as in [1]_.

    Parameters
    ----------
    activations : numpy array
        Beat activation function.
    alpha : float or numpy array
        Scaling factor for the comb filter; if only a single value is given,
        the same scaling factor for all delays is assumed.
    min_tau : int, optional
        Minimal delay for the comb filter [frames].
    max_tau : int, optional
        Maximal delta for comb filter [frames].

    Returns
    -------
    histogram_bins : numpy array
        Bins of the tempo histogram.
    histogram_delays : numpy array
        Corresponding delays [frames].

    References
    ----------
    .. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
           "Accurate Tempo Estimation based on Recurrent Neural Networks and
           Resonating Comb Filters",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    """
    from madmom.audio.comb_filters import CombFilterbankProcessor
    if max_tau is None:
        max_tau = len(activations) - min_tau
    taus = np.arange(min_tau, max_tau + 1)
    cfb = CombFilterbankProcessor('backward', taus, alpha)
    if activations.ndim in (1, 2):
        act = cfb.process(activations)
        act_max = act == np.max(act, axis=-1)[..., np.newaxis]
        histogram_bins = np.sum(act * act_max, axis=0)
    else:
        raise NotImplementedError('too many dimensions for comb filter interval histogram calculation.')
    return (histogram_bins, taus)