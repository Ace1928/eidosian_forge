import math
import warnings
from typing import Optional
import torch
from torchaudio.functional.functional import _create_triangular_filterbank
def barkscale_fbanks(n_freqs: int, f_min: float, f_max: float, n_barks: int, sample_rate: int, bark_scale: str='traunmuller') -> torch.Tensor:
    """Create a frequency bin conversion matrix.

    .. devices:: CPU

    .. properties:: TorchScript

    .. image:: https://download.pytorch.org/torchaudio/doc-assets/bark_fbanks.png
        :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_barks (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        torch.Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_barks``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * barkscale_fbanks(A.size(-1), ...)``.

    """
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = _hz_to_bark(f_min, bark_scale=bark_scale)
    m_max = _hz_to_bark(f_max, bark_scale=bark_scale)
    m_pts = torch.linspace(m_min, m_max, n_barks + 2)
    f_pts = _bark_to_hz(m_pts, bark_scale=bark_scale)
    fb = _create_triangular_filterbank(all_freqs, f_pts)
    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(f'At least one bark filterbank has all zero values. The value for `n_barks` ({n_barks}) may be set too high. Or, the value for `n_freqs` ({n_freqs}) may be set too low.')
    return fb