from __future__ import annotations
import math
import operator
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from functools import reduce
from itertools import chain
from typing import Literal, Optional
import torch
from torch import nn
from tqdm import tqdm
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer
from peft.utils import (
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LoraConfig
from .gptq import dispatch_gptq
from .layer import Conv2d, LoraLayer, dispatch_default
from .tp_layer import dispatch_megatron
def add_weighted_adapter(self, adapters, weights, adapter_name, combination_type='svd', svd_rank=None, svd_clamp=None, svd_full_matrices=True, svd_driver=None, density=None, majority_sign_method: Literal['total', 'frequency']='total') -> None:
    """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                The merging type can be one of [`svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`,
                `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`]. When using the `cat`
                combination_type, the rank of the resulting adapter is equal to the sum of all adapters ranks (the
                mixed adapter may be too big and result in OOM errors).
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
            density (`float`, *optional*):
                Value between 0 and 1. 0 means all values are pruned and 1 means no values are pruned. Should be used
                with [`ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`,
                `magnintude_prune`, `magnitude_prune_svd`]
            majority_sign_method (`str`):
                The method, should be one of ["total", "frequency"], to use to get the magnitude of the sign values.
                Should be used with [`ties`, `ties_svd`, `dare_ties`, `dare_ties_svd`]
        """
    if adapter_name in list(self.peft_config.keys()):
        return
    for adapter in adapters:
        if adapter not in list(self.peft_config.keys()):
            raise ValueError(f'Adapter {adapter} does not exist')
    combination_type = 'linear' if len(adapters) == 1 else combination_type
    adapters_ranks = [self.peft_config[adapter].r for adapter in adapters]
    if combination_type in ('linear', 'ties', 'dare_ties', 'dare_linear', 'magnitude_prune'):
        if len(set(adapters_ranks)) != 1:
            raise ValueError('All adapters must have the same r value when using combination_type linear, ties, dare_ties or dare_linear.')
        new_rank = adapters_ranks[0]
    elif combination_type == 'cat':
        new_rank = sum(adapters_ranks)
    elif combination_type.endswith('svd'):
        new_rank = svd_rank or max(adapters_ranks)
    else:
        raise ValueError(f'Invalid combination_type: {combination_type}')
    target_module_types = [type(self.peft_config[adapter].target_modules) for adapter in adapters]
    if not target_module_types:
        raise ValueError(f'Found no adapter matching the names in {adapters}')
    if len(set(target_module_types)) > 1:
        raise ValueError('all adapter configs should follow the same target modules type. Combining adapters with `target_modules` type being a mix of list/set and string is not supported.')
    if target_module_types[0] == str:
        new_target_modules = '|'.join((f'({self.peft_config[adapter].target_modules})' for adapter in adapters))
    elif target_module_types[0] == set:
        new_target_modules = reduce(operator.or_, (self.peft_config[adapter].target_modules for adapter in adapters))
    else:
        raise TypeError(f'Invalid type {target_module_types[0]} found in target_modules')
    self.peft_config[adapter_name] = replace(self.peft_config[adapters[0]], r=new_rank, lora_alpha=new_rank, target_modules=new_target_modules)
    self.inject_adapter(self.model, adapter_name)
    _freeze_adapter(self.model, adapter_name)
    key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
    for key in key_list:
        _, target, _ = _get_submodules(self.model, key)
        if isinstance(target, LoraLayer):
            if adapter_name in target.lora_A:
                target_lora_A = target.lora_A[adapter_name].weight
                target_lora_B = target.lora_B[adapter_name].weight
            elif adapter_name in target.lora_embedding_A:
                target_lora_A = target.lora_embedding_A[adapter_name]
                target_lora_B = target.lora_embedding_B[adapter_name]
            else:
                continue
            target_lora_A.data = target_lora_A.data * 0.0
            target_lora_B.data = target_lora_B.data * 0.0
            if combination_type == 'cat':
                loras_A, loras_B = ([], [])
                for adapter, weight in zip(adapters, weights):
                    if adapter in target.lora_A:
                        current_adapter_lora_A = target.lora_A[adapter].weight
                        current_adapter_lora_B = target.lora_B[adapter].weight
                    elif adapter in target.lora_embedding_A:
                        current_adapter_lora_A = target.lora_embedding_A[adapter]
                        current_adapter_lora_B = target.lora_embedding_B[adapter]
                    else:
                        continue
                    loras_A.append(current_adapter_lora_A.data * weight * target.scaling[adapter])
                    loras_B.append(current_adapter_lora_B.data)
                if len(loras_A) == 0:
                    raise ValueError('No matching LoRAs found. Please raise an issue on GitHub.')
                loras_A = torch.cat(loras_A, dim=0)
                loras_B = torch.cat(loras_B, dim=1)
                target_lora_A.data[:loras_A.shape[0], :] = loras_A
                target_lora_B.data[:, :loras_B.shape[1]] = loras_B
            elif combination_type in ['svd', 'ties_svd', 'dare_linear_svd', 'dare_ties_svd', 'magnitude_prune_svd']:
                target_lora_A.data, target_lora_B.data = self._svd_generalized_task_arithmetic_weighted_adapter(combination_type, adapters, weights, new_rank, target, target_lora_A, target_lora_B, density, majority_sign_method, svd_clamp, full_matrices=svd_full_matrices, driver=svd_driver)
            elif combination_type in ['linear', 'ties', 'dare_linear', 'dare_ties', 'magnitude_prune']:
                target_lora_A.data, target_lora_B.data = self._generalized_task_arithmetic_weighted_adapter(combination_type, adapters, weights, target, density, majority_sign_method)