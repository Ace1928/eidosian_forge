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
def extract_weights(model_name_a: str, model_a: nn.Module, model_name_b: str, model_b: nn.Module, base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]]=None, unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None, op_to_type_to_weight_extraction_fn: Optional[Dict[str, Dict[Callable, Callable]]]=None) -> NSResultsType:
    """
    Extract weights from model A and model B, and return a comparison.

    Args:
        model_name_a: string name of model A to use in results
        model_a: model A
        model_name_b: string name of model B to use in results
        model_b: model B
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change
        unmatchable_types_map: optional override of unmatchable types, subject to change
        op_to_type_to_weight_extraction_fn: optional override of function which extracts weight
            from a type, subject to change

    Return:
        NSResultsType, containing the weight comparisons
    """
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx.extract_weights')
    if base_name_to_sets_of_related_ops is None:
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    type_a_related_to_b = get_type_a_related_to_b(base_name_to_sets_of_related_ops)
    skipped_module_names: List[str] = []
    skipped_module_classes: List[Callable] = []
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(model_a, 'node_name_to_scope')
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(model_b, 'node_name_to_scope')
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    return _extract_weights_impl(model_name_a, gm_a, model_name_b, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map, op_to_type_to_weight_extraction_fn)