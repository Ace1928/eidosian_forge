import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def equalizer_biquad(waveform: Tensor, sample_rate: int, center_freq: float, gain: float, Q: float=0.707) -> Tensor:
    """Design biquad peaking equalizer filter and perform filtering.  Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        center_freq (float): filter's central frequency
        gain (float or torch.Tensor): desired gain at the boost (or attenuation) in dB
        Q (float or torch.Tensor, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`
    """
    dtype = waveform.dtype
    device = waveform.device
    center_freq = torch.as_tensor(center_freq, dtype=dtype, device=device)
    Q = torch.as_tensor(Q, dtype=dtype, device=device)
    gain = torch.as_tensor(gain, dtype=dtype, device=device)
    w0 = 2 * math.pi * center_freq / sample_rate
    A = torch.exp(gain / 40.0 * math.log(10))
    alpha = torch.sin(w0) / 2 / Q
    b0 = 1 + alpha * A
    b1 = -2 * torch.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha / A
    return biquad(waveform, b0, b1, b2, a0, a1, a2)