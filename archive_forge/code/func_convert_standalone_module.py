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
def convert_standalone_module(node: Node, modules: Dict[str, torch.nn.Module], model: torch.fx.GraphModule, is_reference: bool, backend_config: Optional[BackendConfig]) -> None:
    """ Converts a observed standalone module to a quantized standalone module by calling
    the fx convert api, currently using the same `is_reference` flag as parent, but we may
    changing this behavior in the future (e.g. separating quantization and lowering for
    standalone module as well)

    Args:
      - node: The call_module node of the observed standalone module
      - modules: named_module of original model
      - model: original model
      - is_reference: a flag from parent provided by user to decide if we want to
        produce a reference model or a fbgemm/qnnpack model
      - backend_config: backend configuration of the target backend of quantization
    """
    if is_reference:
        convert_fn = torch.ao.quantization.quantize_fx.convert_to_reference_fx
    else:
        convert_fn = torch.ao.quantization.quantize_fx.convert_fx
    observed_standalone_module: GraphModule = modules[str(node.target)]
    sm_input_quantized_idxs = observed_standalone_module.meta['_observed_graph_module_attrs'].standalone_module_input_quantized_idxs
    args = list(node.args)
    for idx in range(len(args)):
        if idx in sm_input_quantized_idxs:
            arg = args[idx]
            if arg.op == 'call_method' and arg.target == 'dequantize':
                quantize_node = arg.args[0]
                node.replace_input_with(arg, quantize_node)
                if len(arg.users) == 0:
                    model.graph.erase_node(arg)
    sm_output_quantized_idxs = observed_standalone_module.meta['_observed_graph_module_attrs'].standalone_module_output_quantized_idxs
    if len(sm_output_quantized_idxs) > 0:
        assert sm_output_quantized_idxs[0] == 0, 'Currently only quantized'
        'output idxs = [0] is supported'
        _insert_dequantize_node(node, model.graph)
    quantized_standalone_module = convert_fn(observed_standalone_module, backend_config=backend_config)
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, quantized_standalone_module)
    modules[str(node.target)] = quantized_standalone_module