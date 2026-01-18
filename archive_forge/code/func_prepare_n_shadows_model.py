import collections
import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.ns.fx.mappings import (
from torch.ao.ns.fx.graph_matcher import (
from .fx.weight_utils import (
from .fx.graph_passes import (
from .fx.utils import (
from .fx.ns_types import (
from torch.ao.quantization.backend_config.utils import get_fusion_pattern_to_root_node_getter
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.fx.match_utils import _find_matches
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from torch.ao.quantization.fx.qconfig_mapping_utils import _generate_node_name_to_qconfig
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization import QConfigMapping
from torch.ao.ns.fx.n_shadows_utils import (
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping
from typing import Dict, Tuple, Callable, List, Optional, Set, Any, Type
def prepare_n_shadows_model(model: torch.nn.Module, example_inputs: Any, qconfig_multi_mapping: QConfigMultiMapping, backend_config: BackendConfig, custom_prepare_fn: Optional[Callable]=None, custom_prepare_kwargs: Optional[Dict[str, Any]]=None, custom_tracer: Any=None) -> GraphModule:
    """
    Given a model with a graph with M ops such as


      args_kwargs_m -> op_m -> output_m


    And a set of N qconfigs for each op, creates a new model, with
    each of the subgraph of `op_m` transformed into

    .. code::

           |---------> op_m_n -> log_m_n
           |                     /
      args_kwargs_m ---------> op_m -> log_m_0

    Where op_m_n is op_m wrapped in a submodule and transformed with
    qconfig_n, and its inner graph looks like

    .. code::

      args_m -------- op_m_prepared_with_qconfig_n -> out_m_n
                  /
      kwargs_m ---

    This is useful for testing different quantization of multiple layers in
    a single pass through the model.

    High level TODOs for future PRs:
    * figure out a better way to name the output structure
    * return a results data structure instead of printing it out
    * add examples to docblocks
    """
    if custom_tracer is None:
        tracer = quantize_fx.QuantizationTracer([], [])
    else:
        tracer = custom_tracer
    mt = torch.fx.GraphModule(model, tracer.trace(model))
    mt._node_name_to_scope = tracer.node_name_to_scope
    output_prop = OutputProp(mt)
    output_prop.propagate(*example_inputs)
    modules = dict(mt.named_modules(remove_duplicate=False))
    patterns = _get_pattern_to_quantize_handlers(backend_config)
    root_node_getter_mapping = get_fusion_pattern_to_root_node_getter(backend_config)
    standalone_module_names: List[str] = []
    standalone_module_classes: List[Type] = []
    custom_module_classes: List[Type] = []
    matches = _find_matches(mt.graph, modules, patterns, root_node_getter_mapping, standalone_module_names, standalone_module_classes, custom_module_classes)
    subgraphs_dedup: Dict[str, List[Node]] = _get_dedup_subgraphs(matches)
    list_of_node_name_to_qconfig: List[Dict[str, QConfigAny]] = []
    for qconfig_mapping in qconfig_multi_mapping.qconfig_mappings_list:
        node_name_to_qconfig = _generate_node_name_to_qconfig(mt, modules, mt.graph, qconfig_mapping, tracer.node_name_to_scope)
        list_of_node_name_to_qconfig.append(node_name_to_qconfig)
    for subgraph_idx, (match_name, nodes_in_this_subgraph) in enumerate(subgraphs_dedup.items()):
        create_n_transformed_and_logged_copies_of_subgraph(mt, subgraph_idx, match_name, nodes_in_this_subgraph, qconfig_multi_mapping.qconfig_mappings_list, list_of_node_name_to_qconfig, custom_prepare_fn, custom_prepare_kwargs)
    return mt