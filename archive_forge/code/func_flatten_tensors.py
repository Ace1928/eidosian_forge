import contextlib
import functools
import logging
import os
import warnings
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
def flatten_tensors(self, tensors: List[Tensor], aligned_numel: int) -> Tensor:
    """
        Flattens ``tensors`` into a single flat tensor optionally including
        padding if ``aligned_numel`` is greater than 0, where ``aligned_numel``
        gives the numel required to have address alignment.

        NOTE: The padding alignment algorithm must be kept in sync with
        :meth:`_init_flat_param_metadata`. We separate the two methods because
        the initialization happens once, whereas this method may be called
        multiple times throughout training (e.g. for checkpointing).
        """
    if len(tensors) == 0:
        raise ValueError('Expects non-empty `tensors`')
    if aligned_numel < 0:
        raise ValueError(f'Expects non-negative `aligned_numel` but got {aligned_numel}')
    dtype, _, device = self._validate_tensors_to_flatten(tensors)
    flat_tensors: List[Tensor] = []
    if aligned_numel > 0:
        total_numel = 0
        for tensor in tensors:
            numel_to_pad = aligned_numel - total_numel % aligned_numel
            if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                padding_tensor = _construct_padding_tensor(numel_to_pad, dtype, False, device)
                flat_tensors.append(padding_tensor)
                total_numel += numel_to_pad
            flat_tensors.append(torch.flatten(_detach_if_needed(tensor)))
            total_numel += tensor.numel()
        numel_to_pad = self.world_size - total_numel % self.world_size
        if numel_to_pad > 0 and numel_to_pad < self.world_size:
            padding_tensor = _construct_padding_tensor(numel_to_pad, dtype, False, device)
            flat_tensors.append(padding_tensor)
            total_numel += numel_to_pad
    else:
        flat_tensors = [torch.flatten(_detach_if_needed(tensor)) for tensor in tensors]
    return torch.cat(flat_tensors, dim=0)