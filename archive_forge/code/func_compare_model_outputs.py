import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.quantization import prepare
from typing import Dict, List, Optional, Any, Union, Callable, Set
from torch.ao.quantization.quantization_mappings import (
def compare_model_outputs(float_model: nn.Module, q_model: nn.Module, *data, logger_cls=OutputLogger, allow_list=None) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compare output activations between float and quantized models at
    corresponding locations for the same input. Return a dict with key corresponding
    to quantized module names and each entry being a dictionary with two keys
    'float' and 'quantized', containing the activations of quantized model and
    float model at matching locations. This dict can be used to compare and
    compute the propagation quantization error.

    Example usage::

        act_compare_dict = compare_model_outputs(float_model, qmodel, data)
        for key in act_compare_dict:
            print(
                key,
                compute_error(
                    act_compare_dict[key]['float'],
                    act_compare_dict[key]['quantized'].dequantize()
                )
            )

    Args:
        float_model: float model used to generate the q_model
        q_model: model quantized from float_model
        data: input data used to run the prepared float_model and q_model
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger

    Return:
        act_compare_dict: dict with key corresponding to quantized module names
        and each entry being a dictionary with two keys 'float' and 'quantized',
        containing the matching float and quantized activations
    """
    torch._C._log_api_usage_once('quantization_api._numeric_suite.compare_model_outputs')
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()
    prepare_model_outputs(float_model, q_model, logger_cls, allow_list)
    float_model(*data)
    q_model(*data)
    act_compare_dict = get_matching_activations(float_model, q_model)
    return act_compare_dict