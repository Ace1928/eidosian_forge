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
def loudness(waveform: Tensor, sample_rate: int):
    """Measure audio loudness according to the ITU-R BS.1770-4 recommendation.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        waveform(torch.Tensor): audio waveform of dimension `(..., channels, time)`
        sample_rate (int): sampling rate of the waveform

    Returns:
        Tensor: loudness estimates (LKFS)

    Reference:
        - https://www.itu.int/rec/R-REC-BS.1770-4-201510-I/en
    """
    if waveform.size(-2) > 5:
        raise ValueError('Only up to 5 channels are supported.')
    gate_duration = 0.4
    overlap = 0.75
    gamma_abs = -70.0
    kweight_bias = -0.691
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))
    waveform = treble_biquad(waveform, sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
    waveform = highpass_biquad(waveform, sample_rate, 38.0, 0.5)
    energy = torch.square(waveform).unfold(-1, gate_samples, step)
    energy = torch.mean(energy, dim=-1)
    g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)
    g = g[:energy.size(-2)]
    energy_weighted = torch.sum(g.unsqueeze(-1) * energy, dim=-2)
    loudness = -0.691 + 10 * torch.log10(energy_weighted)
    gated_blocks = loudness > gamma_abs
    gated_blocks = gated_blocks.unsqueeze(-2)
    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    gamma_rel = kweight_bias + 10 * torch.log10(energy_weighted) - 10
    gated_blocks = torch.logical_and(gated_blocks.squeeze(-2), loudness > gamma_rel.unsqueeze(-1))
    gated_blocks = gated_blocks.unsqueeze(-2)
    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    LKFS = kweight_bias + 10 * torch.log10(energy_weighted)
    return LKFS