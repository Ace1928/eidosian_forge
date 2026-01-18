from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import signal_frame, smooth as smooth_signal
from ..ml.nn import average_predictions
from ..processors import (OnlineProcessor, ParallelProcessor, Processor,
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