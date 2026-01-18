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
def apply_beamforming(beamform_weights: Tensor, specgram: Tensor) -> Tensor:
    """Apply the beamforming weight to the multi-channel noisy spectrum to obtain the single-channel enhanced spectrum.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    .. math::
        \\hat{\\textbf{S}}(f) = \\textbf{w}_{\\text{bf}}(f)^{\\mathsf{H}} \\textbf{Y}(f)

    where :math:`\\textbf{w}_{\\text{bf}}(f)` is the beamforming weight for the :math:`f`-th frequency bin,
    :math:`\\textbf{Y}` is the multi-channel spectrum for the :math:`f`-th frequency bin.

    Args:
        beamform_weights (Tensor): The complex-valued beamforming weight matrix.
            Tensor of dimension `(..., freq, channel)`
        specgram (Tensor): The multi-channel complex-valued noisy spectrum.
            Tensor of dimension `(..., channel, freq, time)`

    Returns:
        Tensor: The single-channel complex-valued enhanced spectrum.
            Tensor of dimension `(..., freq, time)`
    """
    if beamform_weights.shape[:-2] != specgram.shape[:-3]:
        raise ValueError(f'The dimensions except the last two dimensions of beamform_weights should be the same as the dimensions except the last three dimensions of specgram. Found {beamform_weights.shape} for beamform_weights and {specgram.shape} for specgram.')
    if not (beamform_weights.is_complex() and specgram.is_complex()):
        raise TypeError(f'The type of beamform_weights and specgram must be ``torch.cfloat`` or ``torch.cdouble``. Found {beamform_weights.dtype} for beamform_weights and {specgram.dtype} for specgram.')
    specgram_enhanced = torch.einsum('...fc,...cft->...ft', [beamform_weights.conj(), specgram])
    return specgram_enhanced