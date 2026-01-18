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
def mask_along_axis_iid(specgrams: Tensor, mask_param: int, mask_value: float, axis: int, p: float=1.0) -> Tensor:
    """Apply a mask along ``axis``.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Mask will be applied from indices ``[v_0, v_0 + v)``,
    where ``v`` is sampled from ``uniform(0, max_v)`` and
    ``v_0`` from ``uniform(0, specgrams.size(axis) - v)``,
    with ``max_v = mask_param`` when ``p = 1.0`` and
    ``max_v = min(mask_param, floor(specgrams.size(axis) * p))`` otherwise.

    Args:
        specgrams (Tensor): Real spectrograms `(..., freq, time)`, with at least 3 dimensions.
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on, which should be the one of the last two dimensions.
        p (float, optional): maximum proportion of columns that can be masked. (Default: 1.0)

    Returns:
        Tensor: Masked spectrograms with the same dimensions as input specgrams Tensor`
    """
    dim = specgrams.dim()
    if dim < 3:
        raise ValueError(f'Spectrogram must have at least three dimensions ({dim} given).')
    if axis not in [dim - 2, dim - 1]:
        raise ValueError(f'Only Frequency and Time masking are supported (axis {dim - 2} and axis {dim - 1} supported; {axis} given).')
    if not 0.0 <= p <= 1.0:
        raise ValueError(f'The value of p must be between 0.0 and 1.0 ({p} given).')
    mask_param = _get_mask_param(mask_param, p, specgrams.shape[axis])
    if mask_param < 1:
        return specgrams
    device = specgrams.device
    dtype = specgrams.dtype
    value = torch.rand(specgrams.shape[:dim - 2], device=device, dtype=dtype) * mask_param
    min_value = torch.rand(specgrams.shape[:dim - 2], device=device, dtype=dtype) * (specgrams.size(axis) - value)
    mask_start = min_value.long()[..., None, None]
    mask_end = (min_value.long() + value.long())[..., None, None]
    mask = torch.arange(0, specgrams.size(axis), device=device, dtype=dtype)
    specgrams = specgrams.transpose(axis, -1)
    specgrams = specgrams.masked_fill((mask >= mask_start) & (mask < mask_end), mask_value)
    specgrams = specgrams.transpose(axis, -1)
    return specgrams