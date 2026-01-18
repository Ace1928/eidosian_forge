import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def lowpass_biquad(waveform: Tensor, sample_rate: int, cutoff_freq: float, Q: float=0.707) -> Tensor:
    """Design biquad lowpass filter and perform filtering.  Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        cutoff_freq (float or torch.Tensor): filter cutoff frequency
        Q (float or torch.Tensor, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`
    """
    dtype = waveform.dtype
    device = waveform.device
    cutoff_freq = torch.as_tensor(cutoff_freq, dtype=dtype, device=device)
    Q = torch.as_tensor(Q, dtype=dtype, device=device)
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = torch.sin(w0) / 2 / Q
    b0 = (1 - torch.cos(w0)) / 2
    b1 = 1 - torch.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)