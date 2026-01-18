import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def dither(waveform: Tensor, density_function: str='TPDF', noise_shaping: bool=False) -> Tensor:
    """Apply dither

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Dither increases the perceived dynamic range of audio stored at a
    particular bit-depth by eliminating nonlinear truncation distortion
    (i.e. adding minimally perceived noise to mask distortion caused by quantization).

    Args:
        waveform (Tensor): Tensor of audio of dimension (..., time)
        density_function (str, optional):
            The density function of a continuous random variable. One of
            ``"TPDF"`` (Triangular Probability Density Function),
            ``"RPDF"`` (Rectangular Probability Density Function) or
            ``"GPDF"`` (Gaussian Probability Density Function) (Default: ``"TPDF"``).
        noise_shaping (bool, optional): a filtering process that shapes the spectral
            energy of quantisation error (Default: ``False``)

    Returns:
       Tensor: waveform dithered
    """
    dithered = _apply_probability_distribution(waveform, density_function=density_function)
    if noise_shaping:
        return _add_noise_shaping(dithered, waveform)
    else:
        return dithered