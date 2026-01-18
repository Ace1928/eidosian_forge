import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
from .configuration_patchtsmixer import PatchTSMixerConfig
def forecast_masking(inputs: torch.Tensor, num_forecast_mask_patches: Union[list, int], unmasked_channel_indices: list=None, mask_value: int=0):
    """Forecast masking that masks the last K patches where K is from the num_forecast_mask_patches.
    If num_forecast_mask_patches is a list, samples in the batch will be randomly masked by numbers defined in the list.

    Parameters:
        inputs (`torch.Tensor`):
            Input of shape `(bs, num_channels, num_patch, patch_length)`
        num_forecast_mask_patches (`list`):
            Number of patches to be masked at the end of each batch sample. e.g. 4 or [3, 5].
        unmasked_channel_indices (`list`, *optional*):
            Indices of channels that are not masked.
        mask_value (`int`, *optional*, defaults to 0):
            Values in the masked patches will be filled by `mask_value`.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as inputs Tensor and Mask tensor of shape `(bs,
        num_channels , num_patch)` or `(bs, tsg1, tsg2, num_channels, num_patch)`
    """
    if isinstance(num_forecast_mask_patches, int):
        num_forecast_mask_patches = [num_forecast_mask_patches]
    forecast_mask_ratios = [1 for _ in num_forecast_mask_patches]
    batch_size, num_channels, sequence_length, num_features = inputs.shape
    mask = torch.zeros(batch_size, num_channels, sequence_length, device=inputs.device)
    t_list = []
    total_length = 0
    total_ratio = sum(forecast_mask_ratios)
    for patch_length, ratio in zip(num_forecast_mask_patches, forecast_mask_ratios):
        if patch_length <= 0 or patch_length >= sequence_length:
            raise ValueError(f'num_forecast_mask_patches {patch_length} should be greater than 0 and less than total patches.')
        temp_len = int(batch_size * ratio / total_ratio)
        t_list.append([patch_length, ratio, temp_len])
        total_length += temp_len
    t_list = sorted(t_list, key=lambda x: x[2])
    if total_length < batch_size:
        t_list[0][2] = t_list[0][2] + (batch_size - total_length)
    elif total_length > batch_size:
        t_list[-1][2] = t_list[-1][2] + (total_length - batch_size)
    batch1 = 0
    for patch_len, _, temp_len in t_list:
        batch2 = batch1 + temp_len
        mask[batch1:batch2, :, -patch_len:] = 1
        batch1 = batch2
    perm = torch.randperm(mask.shape[0])
    mask = mask[perm]
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0
    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return (inputs_mask, mask[..., 0])