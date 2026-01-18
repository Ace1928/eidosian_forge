from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
def detect_beats(activations, interval, look_aside=0.2):
    """
    Detects the beats in the given activation function as in [1]_.

    Parameters
    ----------
    activations : numpy array
        Beat activations.
    interval : int
        Look for the next beat each `interval` frames.
    look_aside : float
        Look this fraction of the `interval` to each side to detect the beats.

    Returns
    -------
    numpy array
        Beat positions [frames].

    Notes
    -----
    A Hamming window of 2 * `look_aside` * `interval` is applied around the
    position where the beat is expected to prefer beats closer to the centre.

    References
    ----------
    .. [1] Sebastian Böck and Markus Schedl,
           "Enhanced Beat Tracking with Context-Aware Neural Networks",
           Proceedings of the 14th International Conference on Digital Audio
           Effects (DAFx), 2011.

    """
    sys.setrecursionlimit(len(activations))
    frames_look_aside = max(1, int(interval * look_aside))
    win = np.hamming(2 * frames_look_aside)
    positions = []

    def recursive(position):
        """
        Recursively detect the next beat.

        Parameters
        ----------
        position : int
            Start at this position.

        """
        act = signal_frame(activations, position, frames_look_aside * 2, 1)
        act = np.multiply(act, win)
        if np.argmax(act) > 0:
            position = np.argmax(act) + position - frames_look_aside
        positions.append(position)
        if position + interval < len(activations):
            recursive(position + interval)
        else:
            return
    sums = np.zeros(interval)
    for i in range(interval):
        positions = []
        recursive(i)
        sums[i] = np.sum(activations[positions])
    start_position = np.argmax(sums)
    positions = []
    recursive(start_position)
    return np.array(positions)