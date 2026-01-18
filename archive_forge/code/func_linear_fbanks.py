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
def linear_fbanks(n_freqs: int, f_min: float, f_max: float, n_filter: int, sample_rate: int) -> Tensor:
    """Creates a linear triangular filterbank.

    .. devices:: CPU

    .. properties:: TorchScript

    Note:
        For the sake of the numerical compatibility with librosa, not all the coefficients
        in the resulting filter bank has magnitude of 1.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/lin_fbanks.png
           :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_filter (int): Number of (linear) triangular filter
        sample_rate (int): Sample rate of the audio waveform

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_filter``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * linear_fbanks(A.size(-1), ...)``.
    """
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    f_pts = torch.linspace(f_min, f_max, n_filter + 2)
    fb = _create_triangular_filterbank(all_freqs, f_pts)
    return fb