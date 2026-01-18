import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.quantization import prepare
from typing import Dict, List, Optional, Any, Union, Callable, Set
from torch.ao.quantization.quantization_mappings import (
def compare_weights(float_dict: Dict[str, Any], quantized_dict: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compare the weights of the float module with its corresponding quantized
    module. Return a dict with key corresponding to module names and each entry being
    a dictionary with two keys 'float' and 'quantized', containing the float and
    quantized weights. This dict can be used to compare and compute the quantization
    error of the weights of float and quantized models.

    Example usage::

        wt_compare_dict = compare_weights(
            float_model.state_dict(), qmodel.state_dict())
        for key in wt_compare_dict:
            print(
                key,
                compute_error(
                    wt_compare_dict[key]['float'],
                    wt_compare_dict[key]['quantized'].dequantize()
                )
            )

    Args:
        float_dict: state dict of the float model
        quantized_dict: state dict of the quantized model

    Return:
        weight_dict: dict with key corresponding to module names and each entry being
        a dictionary with two keys 'float' and 'quantized', containing the float and
        quantized weights
    """
    torch._C._log_api_usage_once('quantization_api._numeric_suite.compare_weights')
    weight_dict: Dict[str, Dict] = {}
    for key in quantized_dict:
        match_key = _find_match(float_dict, key, 'weight')
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]['float'] = float_dict[match_key]
            weight_dict[key]['quantized'] = quantized_dict[key]
            continue
        match_key = _find_match(float_dict, key, '_packed_params')
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]['float'] = float_dict[match_key]
            weight_dict[key]['quantized'] = quantized_dict[key][0]
        split_str = key.split('.')
        if split_str[-1] == 'param' and split_str[-3] == '_all_weight_values':
            layer = split_str[-2]
            module_name = '.'.join(split_str[:-3])
            float_weight_ih_key = module_name + '.weight_ih_l' + layer
            float_weight_hh_key = module_name + '.weight_hh_l' + layer
            if float_weight_ih_key in float_dict and float_weight_hh_key in float_dict:
                weight_dict[key] = {}
                weight_dict[key]['float'] = float_dict[float_weight_ih_key]
                weight_dict[key]['quantized'] = quantized_dict[key].__getstate__()[0][4][0].__getstate__()[0][0]
                weight_dict[key]['float'] = float_dict[float_weight_hh_key]
                weight_dict[key]['quantized'] = quantized_dict[key].__getstate__()[0][4][1].__getstate__()[0][0]
    return weight_dict