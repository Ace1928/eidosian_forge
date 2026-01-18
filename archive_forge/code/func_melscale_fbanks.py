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
def melscale_fbanks(n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int, norm: Optional[str]=None, mel_scale: str='htk') -> Tensor:
    """Create a frequency bin conversion matrix.

    .. devices:: CPU

    .. properties:: TorchScript

    Note:
        For the sake of the numerical compatibility with librosa, not all the coefficients
        in the resulting filter bank has magnitude of 1.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/mel_fbanks.png
           :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A @ melscale_fbanks(A.size(-1), ...)``.

    """
    if norm is not None and norm != 'slaney':
        raise ValueError('norm must be one of None or "slaney"')
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)
    fb = _create_triangular_filterbank(all_freqs, f_pts)
    if norm is not None and norm == 'slaney':
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)
    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(f'At least one mel filterbank has all zero values. The value for `n_mels` ({n_mels}) may be set too high. Or, the value for `n_freqs` ({n_freqs}) may be set too low.')
    return fb