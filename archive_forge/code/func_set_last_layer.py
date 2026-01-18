import logging
import os
import types
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional, Union
import torch
from packaging.version import parse
from ..utils import check_if_pytorch_greater, is_accelerate_available, recurse_getattr, recurse_setattr
from .models import BetterTransformerManager
def set_last_layer(model: torch.nn.Module):
    """
    Iterates over the module list containing the `LayerBetterTransformer` modules. Sets the last layer's `is_last_layer`
    attribute to `True`

    Args:
        `model` (`torch.nn.Module`):
            The input converted model
    Raises:
        `NotImplementedError`: Raised if this method fails, in which case the model is not supported.
    """
    dict_named_module = dict(model.named_modules())
    sort_fn = lambda list_modules: [module.__class__.__name__ for module in list_modules]
    modulelist_lengths = []
    for key in dict_named_module.keys():
        if isinstance(dict_named_module[key], torch.nn.ModuleList) and 'encoder' in key and (model.config.model_type not in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM or (model.config.model_type in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM and all((name not in key for name in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM[model.config.model_type])))):
            modulelist_lengths.append((len(dict_named_module[key]), key))
    if len(modulelist_lengths) > 1:
        _, key = max(modulelist_lengths, key=lambda item: item[0])
        largest_module_list = dict_named_module[key]
        for module in largest_module_list[-1].modules():
            if 'LayerBetterTransformer' in module.__class__.__name__:
                setattr(module, 'is_last_layer', True)
                return
    else:
        for key in dict_named_module.keys():
            if isinstance(dict_named_module[key], torch.nn.ModuleList) and all(('LayerBetterTransformer' in module_name for module_name in sort_fn(dict_named_module[key]))):
                setattr(dict_named_module[key][-1], 'is_last_layer', True)
                return
    raise Exception(f'The transformation of the model {model.__class__.__name__} to BetterTransformer failed while it should not. Please fill a bug report or open a PR to support this model at https://github.com/huggingface/optimum/')