import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def overdrive(waveform: Tensor, gain: float=20, colour: float=20) -> Tensor:
    """Apply a overdrive effect to the audio. Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    This effect applies a non linear distortion to the audio signal.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        gain (float, optional): desired gain at the boost (or attenuation) in dB
            Allowed range of values are 0 to 100
        colour (float, optional):  controls the amount of even harmonic content in the over-driven output
            Allowed range of values are 0 to 100

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    Reference:
        - http://sox.sourceforge.net/sox.html
    """
    actual_shape = waveform.shape
    device, dtype = (waveform.device, waveform.dtype)
    waveform = waveform.view(-1, actual_shape[-1])
    gain = _dB2Linear(gain)
    colour = colour / 200
    last_in = torch.zeros(waveform.shape[:-1], dtype=dtype, device=device)
    last_out = torch.zeros(waveform.shape[:-1], dtype=dtype, device=device)
    temp = waveform * gain + colour
    mask1 = temp < -1
    temp[mask1] = torch.tensor(-2.0 / 3.0, dtype=dtype, device=device)
    mask2 = temp > 1
    temp[mask2] = torch.tensor(2.0 / 3.0, dtype=dtype, device=device)
    mask3 = ~mask1 & ~mask2
    temp[mask3] = temp[mask3] - temp[mask3] ** 3 * (1.0 / 3)
    output_waveform = torch.zeros_like(waveform, dtype=dtype, device=device)
    if device == torch.device('cpu'):
        _overdrive_core_loop_cpu(waveform, temp, last_in, last_out, output_waveform)
    else:
        _overdrive_core_loop_generic(waveform, temp, last_in, last_out, output_waveform)
    return output_waveform.clamp(min=-1, max=1).view(actual_shape)