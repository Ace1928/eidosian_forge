import warnings
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from ..observer import _with_args, ObserverBase, PerChannelMinMaxObserver
from ..utils import _parent_name, check_min_max_valid
from .utils import (
def convert_eq_obs(model: GraphModule, modules: Dict[str, nn.Module], weight_eq_obs_dict: Dict[str, _WeightEqualizationObserver]) -> None:
    """ Converts the equalization operations and updates the other nodes in the
    following way:
        - Removes the input equalization observers and inserts a mul operator
          along with an equalization scale node wherever applicable (we do not
          want to insert a mul operator between connecting linear layers).
        - Updates the input quantization observers with the scaled input min/max
          values.
        - Scales the weights by the current and next equalization scales.
        - Removes the weight equalization observer node if it exists.

    Before (after prepare):
                                    weight values
                                          |
                                    WeightQuantObs
                                          |
                                      WeightEqObs
                                          |
        x -> InpQuantObs -> InpEqObs -> linear -> OutQuantObs

    After this function:
                                              scaled weight values
                                                      |
       equalization scale                       WeightQuantObs
              |                                       |
        x -> mul -> InpQuantObs (scaled min/max) -> linear -> OutQuantObs

    After convert:
       equalization scale                 scaled weight values
              |                                    |
        x -> mul -> quantize_per_tensor -> quantized::linear

    Note that although the equalization observer appeared after the quantization
    observer after prepare_fx, the mul node appears before the quantization node
    after convert_fx. This is because placing the equalization observer after
    the quantization observer in prepare_fx would allow us to keep the invariant
    that the graph before the current node inserts its observers is not
    modified.

    Having the equalization observer before the quantization observer would also
    cause some inconsistences between the ordering of the quantization and
    equalization observers.
    For example, a single linear layer would look like:
        x -> InpEqObs1 -> InpQuantObs1 -> linear1 -> OutQuantObs1
    But between two connected linear layers, it would look like:
        linear1 -> OutQuantObs1 -> InpEqObs2 -> linear2 -> OutQuantObs2
    """
    for node in model.graph.nodes:
        if node.op == 'call_module' and isinstance(modules[node.target], _InputEqualizationObserver):
            inp_quant_obs_node = node.args[0]
            prev_node = inp_quant_obs_node.args[0]
            if node_supports_equalization(prev_node, modules) or 'relu' in prev_node.name:
                remove_node(model, node, inp_quant_obs_node)
                continue
            scale_input_observer(node, modules)
            with model.graph.inserting_before(inp_quant_obs_node):
                get_new_eq_scale_name = get_new_attr_name_with_prefix(prev_node.name + '_equalization_scale')
                name = get_new_eq_scale_name(modules)
                setattr(model, name, modules[node.target].equalization_scale)
                eq_scale_node = model.graph.create_node('get_attr', name)
            with model.graph.inserting_after(eq_scale_node):
                inputs = (prev_node, eq_scale_node)
                mul_node = model.graph.create_node('call_function', torch.mul, inputs)
            inp_quant_obs_node.replace_input_with(prev_node, mul_node)
            remove_node(model, node, inp_quant_obs_node)
        elif weight_eq_obs_dict.get(node.name, None) is not None:
            weight_eq_obs = weight_eq_obs_dict.get(node.name)
            assert isinstance(weight_eq_obs, _WeightEqualizationObserver)
            equalization_scale = weight_eq_obs.equalization_scale
            if equalization_scale.nelement() == 1 and equalization_scale == torch.tensor(1):
                equalization_scale = None
            maybe_next_equalization_scale = maybe_get_next_equalization_scale(node, modules)
            if node.op == 'call_module':
                scale_weight_node(node, modules, equalization_scale, maybe_next_equalization_scale)
            elif node.op == 'call_function':
                scale_weight_functional(node, model, modules, equalization_scale, maybe_next_equalization_scale)
                weight_eq_obs_node = maybe_get_weight_eq_obs_node(node, modules)
                if weight_eq_obs_node is None:
                    return
                assert isinstance(modules[str(weight_eq_obs_node.target)], _WeightEqualizationObserver)
                clear_weight_quant_obs_node(node, modules)
                prev_node = weight_eq_obs_node.args[0]
                remove_node(model, weight_eq_obs_node, prev_node)
            else:
                raise ValueError("Expected operation node to be 'call_module' or 'call_function" + f"Instead got node {node.name} as '{node.op}'.")