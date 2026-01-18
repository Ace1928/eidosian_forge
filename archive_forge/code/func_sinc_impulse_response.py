import warnings
from typing import List, Optional, Union
import torch
from torchaudio.functional import fftconvolve
def sinc_impulse_response(cutoff: torch.Tensor, window_size: int=513, high_pass: bool=False):
    """Create windowed-sinc impulse response for given cutoff frequencies.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        cutoff (Tensor): Cutoff frequencies for low-pass sinc filter.

        window_size (int, optional): Size of the Hamming window to apply. Must be odd.
        (Default: 513)

        high_pass (bool, optional):
            If ``True``, convert the resulting filter to high-pass.
            Otherwise low-pass filter is returned. Default: ``False``.

    Returns:
        Tensor: A series of impulse responses. Shape: `(..., window_size)`.
    """
    if window_size % 2 == 0:
        raise ValueError(f'`window_size` must be odd. Given: {window_size}')
    half = window_size // 2
    device, dtype = (cutoff.device, cutoff.dtype)
    idx = torch.linspace(-half, half, window_size, device=device, dtype=dtype)
    filt = torch.special.sinc(cutoff.unsqueeze(-1) * idx.unsqueeze(0))
    filt = filt * torch.hamming_window(window_size, device=device, dtype=dtype, periodic=False).unsqueeze(0)
    filt = filt / filt.sum(dim=-1, keepdim=True).abs()
    if high_pass:
        filt = -filt
        filt[..., half] = 1.0 + filt[..., half]
    return filt