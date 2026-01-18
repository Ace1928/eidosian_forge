import logging
import os
import types
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional, Union
import torch
from packaging.version import parse
from ..utils import check_if_pytorch_greater, is_accelerate_available, recurse_getattr, recurse_setattr
from .models import BetterTransformerManager
def replace_to_bettertransformer(model, config):
    """
    Replaces the current model to its `BetterTransformer` implementation. Loops recursively into the model and replaces the
    `Layer` modules with its `BetterTransformer` correspondant model

    - Step 1: Recurse over the modules of the model
    - Step 2: Verify if the module `BetterTransformer` is present for that model
    - Step 3: If yes, replace the `...Layer` module with the `...LayerBetterTransformer` modules
    - Step 4: If not, yield an error.
    - Step 5: Post process the potentially converted model by setting the `is_last_layer` attribute to `True` for the last `BetterTransformer` layer.
    (done in `set_last_layer` function)

    Args:
        `model` (`torch.nn.Module`):
            The input model to convert
        `config` (`transformers.PreTrainedConfig`):
            The configuration dictionary of the model
    Returns:
        The converted model
    """
    for name, module in model.named_children():
        if hasattr(module, 'SCB'):
            raise ValueError('`load_in_8bit` and `BetterTransformers` are mutually exclusive', ' please pass a model that is not loaded in 8-bit.')
        target_classes = list(BetterTransformerManager.MODEL_MAPPING[config.model_type].keys())
        if config.model_type in BetterTransformerManager.OVERWRITE_METHODS:
            for class_name, method_name_and_replacement in BetterTransformerManager.OVERWRITE_METHODS[config.model_type].items():
                if module.__class__.__name__ == class_name:
                    method_name = method_name_and_replacement[0]
                    new_method = method_name_and_replacement[1]
                    setattr(module, method_name, types.MethodType(new_method, module))
        should_replace_module = False
        for target_class in target_classes:
            should_replace_module = module.__class__.__name__ == target_class
            if should_replace_module:
                bettertransformer_module = BetterTransformerManager.MODEL_MAPPING[config.model_type][target_class](module, config)
                model._modules[name] = bettertransformer_module
                break
        if len(list(module.children())) > 0 and should_replace_module is False:
            if config.model_type not in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM or (config.model_type in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM and name not in BetterTransformerManager.EXCLUDE_FROM_TRANSFORM[config.model_type]):
                replace_to_bettertransformer(module, config)
    return model