import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.quantization import prepare
from typing import Dict, List, Optional, Any, Union, Callable, Set
from torch.ao.quantization.quantization_mappings import (
def get_matching_activations(float_module: nn.Module, q_module: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """Find the matching activation between float and quantized modules.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module

    Return:
        act_dict: dict with key corresponding to quantized module names and each
        entry being a dictionary with two keys 'float' and 'quantized', containing
        the matching float and quantized activations
    """
    torch._C._log_api_usage_once('quantization_api._numeric_suite.get_matching_activations')
    float_dict = get_logger_dict(float_module)
    quantized_dict = get_logger_dict(q_module)
    act_dict: Dict[str, Dict] = {}
    for key in quantized_dict:
        if len(quantized_dict[key]['tensor_val']) == 0:
            continue
        match_key = _find_match(sorted(float_dict, reverse=True), key, 'stats')
        if match_key is not None:
            act_dict[key] = {}
            act_dict[key]['float'] = float_dict[match_key]['tensor_val']
            act_dict[key]['quantized'] = quantized_dict[key]['tensor_val']
    return act_dict