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
def extract_logger_info(model_a: nn.Module, model_b: nn.Module, logger_cls: Callable, model_name_to_use_for_layer_names: str) -> NSResultsType:
    """
    Traverse all loggers in `model_a` and `model_b`, and extract the logged
    information.

    Args:
        model_a: model A
        model_b: model B
        logger_cls: class of Logger to use
        model_name_to_use_for_layer_names: string name of model to use for
          layer names in the output

    Return:
        NSResultsType, containing the logged comparisons
    """
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx.extract_logger_info')
    results: NSResultsType = {}
    for model in (model_a, model_b):
        _extract_logger_info_one_model(model, results, logger_cls)
    maybe_add_missing_fqns(results)
    results = rekey_logger_info_on_node_name_of_model(results, model_name_to_use_for_layer_names)
    return results