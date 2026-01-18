import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def bandpass_biquad(waveform: Tensor, sample_rate: int, central_freq: float, Q: float=0.707, const_skirt_gain: bool=False) -> Tensor:
    """Design two-pole band-pass filter.  Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        central_freq (float or torch.Tensor): central frequency (in Hz)
        Q (float or torch.Tensor, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)
        const_skirt_gain (bool, optional) : If ``True``, uses a constant skirt gain (peak gain = Q).
            If ``False``, uses a constant 0dB peak gain. (Default: ``False``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    Reference:
        - http://sox.sourceforge.net/sox.html
        - https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """
    dtype = waveform.dtype
    device = waveform.device
    central_freq = torch.as_tensor(central_freq, dtype=dtype, device=device)
    Q = torch.as_tensor(Q, dtype=dtype, device=device)
    w0 = 2 * math.pi * central_freq / sample_rate
    alpha = torch.sin(w0) / 2 / Q
    temp = torch.sin(w0) / 2 if const_skirt_gain else alpha
    b0 = temp
    b1 = 0.0
    b2 = -temp
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)