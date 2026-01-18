import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def deemph_biquad(waveform: Tensor, sample_rate: int) -> Tensor:
    """Apply ISO 908 CD de-emphasis (shelving) IIR filter.  Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, Allowed sample rate ``44100`` or ``48000``

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    Reference:
        - http://sox.sourceforge.net/sox.html
        - https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """
    if sample_rate == 44100:
        central_freq = 5283
        width_slope = 0.4845
        gain = -9.477
    elif sample_rate == 48000:
        central_freq = 5356
        width_slope = 0.479
        gain = -9.62
    else:
        raise ValueError('Sample rate must be 44100 (audio-CD) or 48000 (DAT)')
    w0 = 2 * math.pi * central_freq / sample_rate
    A = math.exp(gain / 40.0 * math.log(10))
    alpha = math.sin(w0) / 2 * math.sqrt((A + 1 / A) * (1 / width_slope - 1) + 2)
    temp1 = 2 * math.sqrt(A) * alpha
    temp2 = (A - 1) * math.cos(w0)
    temp3 = (A + 1) * math.cos(w0)
    b0 = A * (A + 1 + temp2 + temp1)
    b1 = -2 * A * (A - 1 + temp3)
    b2 = A * (A + 1 + temp2 - temp1)
    a0 = A + 1 - temp2 + temp1
    a1 = 2 * (A - 1 - temp3)
    a2 = A + 1 - temp2 - temp1
    return biquad(waveform, b0, b1, b2, a0, a1, a2)