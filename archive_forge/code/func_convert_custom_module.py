from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, Callable
from torch.ao.quantization.quant_type import QuantType
import torch
import copy
import warnings
from torch.fx import (
from torch.fx.graph import (
from ..utils import (
from ..qconfig import (
from ..qconfig_mapping import QConfigMapping
from .qconfig_mapping_utils import (
from torch.ao.quantization.backend_config.utils import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.observer import _is_activation_post_process
from .graph_module import (
from ._equalize import update_obs_for_equalization, convert_eq_obs
from torch.nn.utils.parametrize import type_before_parametrizations
from .utils import (
from torch.ao.quantization.utils import (
from torch.ao.quantization.quantize import (
from torch.ao.quantization.stubs import DeQuantStub
from .custom_config import (
from .lower_to_fbgemm import lower_to_fbgemm
from ._decomposed import quantized_decomposed_lib  # noqa: F401
import operator
def convert_custom_module(node: Node, graph: Graph, modules: Dict[str, torch.nn.Module], custom_module_class_mapping: Dict[QuantType, Dict[Type, Type]], statically_quantized_custom_module_nodes: Set[Node]) -> None:
    """ Converts an observed custom module to a quantized custom module based on
    `custom_module_class_mapping`
    For static quantization, we'll also remove the previous `dequantize` node and
    attach the observer node for output to the module, the observer for the node
    will be converted to a dequantize node instead of quantize-dequantize pairs
    later in the graph. In the end we would have a quantized custom module that
    has the same interface as a default quantized module in nn.quantized namespace,
    i.e. quantized input and quantized output.

    Args:
      - node: The call_module node of the observed standalone module
      - graph: The graph containing the node
      - modules: named_module of original model
      - custom_module_class_mapping: mapping from observed custom module class to
        quantized custom module class, used to swap custom modules
      - statically_quantized_custom_module_nodes: we'll add the custom module node
        if we find it is statically quantized, this will be used later when converting
        observers to quant/dequant node pairs, if the observed node is a statically
        quantized custom module nodes, we'll convert the observer to a dequantize node,
        this is to keep the interface the same as the default quantized module.
        TODO: maybe we want to redesign this part to align with reference model design
        as well, but there has been some discussions around the interface, so we can do
        it later.
    """
    observed_custom_module = modules[str(node.target)]
    maybe_obs = _maybe_get_observer_for_node(node, modules)
    qconfig = observed_custom_module.qconfig
    if activation_is_statically_quantized(qconfig):
        statically_quantized_custom_module_nodes.add(node)
        if _is_custom_module_lstm(node, modules):
            assert len(node.args) == 2 and isinstance(node.args[1], tuple) and (len(node.args[1]) == 2)
            inputs, (hidden0, hidden1) = node.args
            assert isinstance(inputs, Node)
            assert isinstance(hidden0, Node)
            assert isinstance(hidden1, Node)
            _remove_previous_dequantize_in_custom_module(node, inputs, graph)
            _remove_previous_dequantize_in_custom_module(node, hidden0, graph)
            _remove_previous_dequantize_in_custom_module(node, hidden1, graph)
        elif _is_custom_module_mha(node, modules):
            assert len(node.args) == 3
            query, key, value = node.args
            assert isinstance(query, Node)
            assert isinstance(key, Node)
            assert isinstance(value, Node)
            _remove_previous_dequantize_in_custom_module(node, query, graph)
            _remove_previous_dequantize_in_custom_module(node, key, graph)
            _remove_previous_dequantize_in_custom_module(node, value, graph)
        else:
            arg = node.args[0]
            assert isinstance(arg, Node)
            _remove_previous_dequantize_in_custom_module(node, arg, graph)
            activation_post_process = _maybe_get_observer_for_node(node, modules)
            assert activation_post_process is not None
            observed_custom_module.activation_post_process = activation_post_process
    quantized_custom_module_class = get_swapped_custom_module_class(observed_custom_module, custom_module_class_mapping, qconfig)
    quantized_custom_module = quantized_custom_module_class.from_observed(observed_custom_module)
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, quantized_custom_module)