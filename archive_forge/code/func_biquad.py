import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def biquad(waveform: Tensor, b0: float, b1: float, b2: float, a0: float, a1: float, a2: float) -> Tensor:
    """Perform a biquad filter of input tensor.  Initial conditions set to 0.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        b0 (float or torch.Tensor): numerator coefficient of current input, x[n]
        b1 (float or torch.Tensor): numerator coefficient of input one time step ago x[n-1]
        b2 (float or torch.Tensor): numerator coefficient of input two time steps ago x[n-2]
        a0 (float or torch.Tensor): denominator coefficient of current output y[n], typically 1
        a1 (float or torch.Tensor): denominator coefficient of current output y[n-1]
        a2 (float or torch.Tensor): denominator coefficient of current output y[n-2]

    Returns:
        Tensor: Waveform with dimension of `(..., time)`

    Reference:
       - https://en.wikipedia.org/wiki/Digital_biquad_filter
    """
    device = waveform.device
    dtype = waveform.dtype
    b0 = torch.as_tensor(b0, dtype=dtype, device=device).view(1)
    b1 = torch.as_tensor(b1, dtype=dtype, device=device).view(1)
    b2 = torch.as_tensor(b2, dtype=dtype, device=device).view(1)
    a0 = torch.as_tensor(a0, dtype=dtype, device=device).view(1)
    a1 = torch.as_tensor(a1, dtype=dtype, device=device).view(1)
    a2 = torch.as_tensor(a2, dtype=dtype, device=device).view(1)
    output_waveform = lfilter(waveform, torch.cat([a0, a1, a2]), torch.cat([b0, b1, b2]))
    return output_waveform