import torch
import torch.fx
from torch.fx import (
from torch.ao.ns.fx.utils import (
from torch.ao.ns.fx.ns_types import (
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.utils._pytree import tree_map
import collections
import copy
from typing import List, Dict, Set, Tuple, Callable, Any, Optional
import operator
def create_n_transformed_and_logged_copies_of_subgraph(mt: GraphModule, subgraph_idx: int, match_name: str, nodes_in_this_subgraph: List[Any], qconfig_mappings: List[QConfigMapping], list_of_node_name_to_qconfig: List[Dict[str, QConfigAny]], custom_prepare_fn: Optional[Callable]=None, custom_prepare_kwargs: Optional[Dict[str, Any]]=None) -> None:
    """
    Given a model `mt` and a subgraph_idx, creates the needed copies
    of the subgraph for all qconfigs, and instruments them with loggers.
    """
    if any((not isinstance(node, Node) for node in nodes_in_this_subgraph)):
        return
    first_node = nodes_in_this_subgraph[0]
    last_node = nodes_in_this_subgraph[-1]
    prev_node = get_normalized_nth_input(first_node, mt, 0)
    if isinstance(prev_node, list):
        example_inputs = [x.traced_result for x in prev_node]
    elif isinstance(prev_node, tuple):
        example_inputs = (x.traced_result for x in prev_node)
    elif hasattr(prev_node, 'traced_result'):
        example_inputs = (prev_node.traced_result,)
    else:
        print('unable to get example input for node ' + f'{first_node.format_node()}, skipping')
        return
    found_at_least_one_qconfig = False
    for subgraph_candidate_idx in range(len(qconfig_mappings) + 1):
        if subgraph_candidate_idx == 0:
            continue
        node_name_to_qconfig = list_of_node_name_to_qconfig[subgraph_candidate_idx - 1]
        qconfig = node_name_to_qconfig[first_node.name]
        if qconfig is not None:
            found_at_least_one_qconfig = True
            break
    if not found_at_least_one_qconfig:
        print('unable to find at least one qconfig for node ' + f'{first_node.format_node()}, skipping')
        return
    fqn = _maybe_get_fqn(first_node, mt)
    last_added_shadow_node_list: List[Optional[Node]] = [None]
    for subgraph_candidate_idx in range(len(qconfig_mappings) + 1):
        create_one_transformed_and_logged_copy_of_subgraph(mt, subgraph_idx, subgraph_candidate_idx, first_node, last_node, fqn, list_of_node_name_to_qconfig, example_inputs, last_added_shadow_node_list, custom_prepare_fn, custom_prepare_kwargs)