import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.quantization import prepare
from typing import Dict, List, Optional, Any, Union, Callable, Set
from torch.ao.quantization.quantization_mappings import (
def prepare_model_with_stubs(float_module: nn.Module, q_module: nn.Module, module_swap_list: Set[type], logger_cls: Callable) -> None:
    """Prepare the model by attaching the float module to its matching quantized
    module as the shadow if the float module type is in module_swap_list.

    Example usage::

        prepare_model_with_stubs(float_model, q_model, module_swap_list, Logger)
        q_model(data)
        ob_dict = get_logger_dict(q_model)

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        module_swap_list: list of float module types to attach the shadow
        logger_cls: type of logger to be used in shadow module to process the outputs of
            quantized module and its float shadow module
    """
    torch._C._log_api_usage_once('quantization_api._numeric_suite.prepare_model_with_stubs')
    float_module_children = {}
    for name, mod in float_module.named_children():
        float_module_children[name] = mod
    reassign = {}
    for name, mod in q_module.named_children():
        if name not in float_module_children:
            continue
        float_mod = float_module_children[name]
        if type(float_mod) not in module_swap_list:
            prepare_model_with_stubs(float_mod, mod, module_swap_list, logger_cls)
        if type(float_mod) in module_swap_list and (not _is_identical_module_type(mod, float_mod)):
            reassign[name] = Shadow(mod, float_mod, logger_cls)
    for key, value in reassign.items():
        q_module._modules[key] = value