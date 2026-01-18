import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated
from .filtering import highpass_biquad, treble_biquad
def compute_deltas(specgram: Tensor, win_length: int=5, mode: str='replicate') -> Tensor:
    """Compute delta coefficients of a tensor, usually a spectrogram:

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    .. math::
       d_t = \\frac{\\sum_{n=1}^{\\text{N}} n (c_{t+n} - c_{t-n})}{2 \\sum_{n=1}^{\\text{N}} n^2}

    where :math:`d_t` is the deltas at time :math:`t`,
    :math:`c_t` is the spectrogram coeffcients at time :math:`t`,
    :math:`N` is ``(win_length-1)//2``.

    Args:
        specgram (Tensor): Tensor of audio of dimension `(..., freq, time)`
        win_length (int, optional): The window length used for computing delta (Default: ``5``)
        mode (str, optional): Mode parameter passed to padding (Default: ``"replicate"``)

    Returns:
        Tensor: Tensor of deltas of dimension `(..., freq, time)`

    Example
        >>> specgram = torch.randn(1, 40, 1000)
        >>> delta = compute_deltas(specgram)
        >>> delta2 = compute_deltas(delta)
    """
    device = specgram.device
    dtype = specgram.dtype
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])
    if win_length < 3:
        raise ValueError(f'Window length should be greater than or equal to 3. Found win_length {win_length}')
    n = (win_length - 1) // 2
    denom = n * (n + 1) * (2 * n + 1) / 3
    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)
    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(specgram.shape[1], 1, 1)
    output = torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom
    output = output.reshape(shape)
    return output