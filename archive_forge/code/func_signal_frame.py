from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from ..processors import BufferProcessor, Processor
from ..utils import integer_types
def signal_frame(signal, index, frame_size, hop_size, origin=0):
    """
    This function returns frame at `index` of the `signal`.

    Parameters
    ----------
    signal : numpy array
        Signal.
    index : int
        Index of the frame to return.
    frame_size : int
        Size of each frame in samples.
    hop_size : float
        Hop size in samples between adjacent frames.
    origin : int
        Location of the window center relative to the signal position.

    Returns
    -------
    frame : numpy array
        Requested frame of the signal.

    Notes
    -----
    The reference sample of the first frame (index == 0) refers to the first
    sample of the `signal`, and each following frame is placed `hop_size`
    samples after the previous one.

    The window is always centered around this reference sample. Its location
    relative to the reference sample can be set with the `origin` parameter.
    Arbitrary integer values can be given:

    - zero centers the window on its reference sample
    - negative values shift the window to the right
    - positive values shift the window to the left

    An `origin` of half the size of the `frame_size` results in windows located
    to the left of the reference sample, i.e. the first frame starts at the
    first sample of the signal.

    The part of the frame which is not covered by the signal is padded with
    zeros.

    This function is totally independent of the length of the signal. Thus,
    contrary to common indexing, the index '-1' refers NOT to the last frame
    of the signal, but instead the frame left of the first frame is returned.

    """
    frame_size = int(frame_size)
    num_samples = len(signal)
    ref_sample = int(index * hop_size)
    start = ref_sample - frame_size // 2 - int(origin)
    stop = start + frame_size
    if stop < 0 or start > num_samples:
        frame = np.repeat(signal[:1] * 0, frame_size, axis=0)
        return frame
    elif start < 0 and stop > num_samples:
        frame = np.repeat(signal[:1] * 0, frame_size, axis=0)
        frame[-start:num_samples - start] = signal
        return frame
    elif start < 0:
        frame = np.repeat(signal[:1] * 0, frame_size, axis=0)
        frame[-start:] = signal[:stop,]
        return frame
    elif stop > num_samples:
        frame = np.repeat(signal[:1] * 0, frame_size, axis=0)
        frame[:num_samples - start] = signal[start:,]
        return frame
    return signal[start:stop,]