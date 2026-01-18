import enum
import operator
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq
from typing import Tuple, Callable, Dict, Set, List, Optional, Union
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization import (
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.observer import _is_activation_post_process
from .ns_types import NSNodeTargetType, NSResultsType
def rekey_logger_info_on_node_name_of_model(results: NSResultsType, model_name: str) -> NSResultsType:
    """
    Rekeys the layer name of a results dictionary to use node names
    from `model_name`.

    For example, transforms

        {'base_op_1_0': {'node_output': {'model_a':
          [{'ref_node_name': 'linear1', ...}]}}}

    into

        {'linear1': {'node_output': {'model_a':
          [{'ref_node_name': 'linear1', ...}]}}}

    Note: we cannot use these node names directly because they are not
    guaranteed to be consistent across models. This is why we extract
    the results first and rekey afterwards.
    """
    new_results = {}
    for old_layer_name, result_type_to_results in results.items():
        new_layer_name = None
        for model_name_to_results in result_type_to_results.values():
            for cur_model_name, list_of_results in model_name_to_results.items():
                if cur_model_name == model_name:
                    assert len(list_of_results)
                    new_layer_name = list_of_results[0]['ref_node_name']
                else:
                    continue
        if new_layer_name is not None:
            new_results[new_layer_name] = result_type_to_results
        else:
            new_results[old_layer_name] = result_type_to_results
    return new_results