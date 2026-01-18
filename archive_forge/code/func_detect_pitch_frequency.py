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
def detect_pitch_frequency(waveform: Tensor, sample_rate: int, frame_time: float=10 ** (-2), win_length: int=30, freq_low: int=85, freq_high: int=3400) -> Tensor:
    """Detect pitch frequency.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    It is implemented using normalized cross-correlation function and median smoothing.

    Args:
        waveform (Tensor): Tensor of audio of dimension `(..., freq, time)`
        sample_rate (int): The sample rate of the waveform (Hz)
        frame_time (float, optional): Duration of a frame (Default: ``10 ** (-2)``).
        win_length (int, optional): The window length for median smoothing (in number of frames) (Default: ``30``).
        freq_low (int, optional): Lowest frequency that can be detected (Hz) (Default: ``85``).
        freq_high (int, optional): Highest frequency that can be detected (Hz) (Default: ``3400``).

    Returns:
        Tensor: Tensor of freq of dimension `(..., frame)`
    """
    shape = list(waveform.size())
    waveform = waveform.reshape([-1] + shape[-1:])
    nccf = _compute_nccf(waveform, sample_rate, frame_time, freq_low)
    indices = _find_max_per_frame(nccf, sample_rate, freq_high)
    indices = _median_smoothing(indices, win_length)
    EPSILON = 10 ** (-9)
    freq = sample_rate / (EPSILON + indices.to(torch.float))
    freq = freq.reshape(shape[:-1] + list(freq.shape[-1:]))
    return freq