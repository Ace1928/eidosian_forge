import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def dcshift(waveform: Tensor, shift: float, limiter_gain: Optional[float]=None) -> Tensor:
    """Apply a DC shift to the audio. Similar to SoX implementation.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    This can be useful to remove a DC offset
    (caused perhaps by a hardware problem in the recording chain) from the audio

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        shift (float): indicates the amount to shift the audio
            Allowed range of values for shift : -2.0 to +2.0
        limiter_gain (float of None, optional): It is used only on peaks to prevent clipping
            It should have a value much less than 1 (e.g. 0.05 or 0.02)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    Reference:
        - http://sox.sourceforge.net/sox.html
    """
    output_waveform = waveform
    limiter_threshold = 0.0
    if limiter_gain is not None:
        limiter_threshold = 1.0 - (abs(shift) - limiter_gain)
    if limiter_gain is not None and shift > 0:
        mask = waveform > limiter_threshold
        temp = (waveform[mask] - limiter_threshold) * limiter_gain / (1 - limiter_threshold)
        output_waveform[mask] = (temp + limiter_threshold + shift).clamp(max=limiter_threshold)
        output_waveform[~mask] = (waveform[~mask] + shift).clamp(min=-1, max=1)
    elif limiter_gain is not None and shift < 0:
        mask = waveform < -limiter_threshold
        temp = (waveform[mask] + limiter_threshold) * limiter_gain / (1 - limiter_threshold)
        output_waveform[mask] = (temp - limiter_threshold + shift).clamp(min=-limiter_threshold)
        output_waveform[~mask] = (waveform[~mask] + shift).clamp(min=-1, max=1)
    else:
        output_waveform = (waveform + shift).clamp(min=-1, max=1)
    return output_waveform