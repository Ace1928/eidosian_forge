import warnings
from typing import List, Optional, Union
import torch
from torchaudio.functional import fftconvolve
def frequency_impulse_response(magnitudes):
    """Create filter from desired frequency response

    Args:
        magnitudes: The desired frequency responses. Shape: `(..., num_fft_bins)`

    Returns:
        Tensor: Impulse response. Shape `(..., 2 * (num_fft_bins - 1))`
    """
    if magnitudes.min() < 0.0:
        warnings.warn('The input frequency response should not contain negative values.')
    ir = torch.fft.fftshift(torch.fft.irfft(magnitudes), dim=-1)
    device, dtype = (magnitudes.device, magnitudes.dtype)
    window = torch.hann_window(ir.size(-1), periodic=False, device=device, dtype=dtype).expand_as(ir)
    return ir * window