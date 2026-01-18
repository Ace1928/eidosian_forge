import collections
import enum
import torch
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from torch.ao.quantization.utils import getattr_from_fqn
from .ns_types import NSSubgraph, NSNodeTargetType
from .mappings import (
from .pattern_utils import (
from torch.ao.quantization import (
from typing import Dict, Tuple, List, Optional, Set, Any
def get_matching_subgraph_pairs(gm_a: GraphModule, gm_b: GraphModule, base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]]=None, unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None) -> Dict[str, Tuple[NSSubgraph, NSSubgraph]]:
    """
    Matches matchable subgraphs of graph_a to graph_b.

    For a node, "matchable" is defined as a node which is not an observer,
    fake_quants, quant or dequant.

    A subgraph can contain one or more nodes.  A subgraph is matchable if
    at least one node inside of it is matchable.  Currently, all nodes in
    a subgraph must be matchable (because we assume no observers will be
    inserted in the middle of a fusion).

    A subgraph is defined by (start_node, end_node).  We assume that only
    start_node and end_node are linked with the surrounding graph, all other
    nodes in a subgraph are self-contained.

    A pair of nodes is "related" if both nodes represent the same mathematical
    operation across different quantization flavors. For example,
    `F.linear` and `torch.ops.quantized.linear` are related, and
    `F.linear` and `torch.nn.Conv` are not related.

    For each matchable pair of nodes node_a and node_b, they will match
    if node_a and node_b are related.

    For graphs A and B, they will match iff:
    1. the number of matchable subgraphs in A and B is equivalent
    2. when iterating through the matchable subgraphs of A and B in the same order, each
       corresponding pair of base nodes is related.

    This enables us to find the corresponding subgraphs between
    graphs of related models.  For example, if we had two graphs such as:

    graph_a: x0 -> conv_0 (type: nn.Conv2d) -> obs_0 -> x1
             w  -/
             b  -/

    graph_b: x0 -> quant_0 -> qconv_0 (type: nnq.Conv2d) -> dequant_0 -> x1
           packed_params_0 -/

    This function will return the following result:
    {
        'conv_0': (  # the name of the node in graph_b
          (conv_0, conv_0),  # (start_node_a, end_node_a)
          (qconv_0, qconv_0),  # (start_node_b, end_node_b)
        ),
    }

    Or, if we have a fusion pattern,

    graph_a: x0 -> linear_0 -> relu_0 -> obs_0 -> x1
             w  -/
             b  -/

    graph_b: x0 -> quant_0 -> linear_relu_0 -> dequant_0 -> x1
           packed_params_0 -/

    This function will return the following result:
    {
        'linear_relu_0': (  # the name of the node in graph_b
          (linear_0, relu_0),  # (start_node_a, end_node_a)
          (linear_relu_0, linear_relu_0),  # (start_node_b, end_node_b)
        ),
    }
    """
    if unmatchable_types_map is None:
        unmatchable_types_map = get_unmatchable_types_map()
    non_matchable_functions = unmatchable_types_map['funs_unmatchable']
    non_matchable_modules = unmatchable_types_map['mods_unmatchable']
    non_matchable_methods = unmatchable_types_map['meths_unmatchable']
    graph_a_iterator = _NSGraphMatchableSubgraphsIterator(gm_a, non_matchable_functions, non_matchable_modules, non_matchable_methods)
    graph_b_iterator = _NSGraphMatchableSubgraphsIterator(gm_b, non_matchable_functions, non_matchable_modules, non_matchable_methods)
    results = collections.OrderedDict()
    if base_name_to_sets_of_related_ops is None:
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    type_a_related_to_b = get_type_a_related_to_b(base_name_to_sets_of_related_ops)
    existing_names_a: Set[str] = set()
    existing_names_b: Set[str] = set()
    while True:
        cur_subgraph_a, cur_subgraph_b = (None, None)
        try:
            cur_subgraph_a = next(graph_a_iterator)
        except StopIteration:
            pass
        try:
            cur_subgraph_b = next(graph_b_iterator)
        except StopIteration:
            pass
        type_start_a, type_start_b = (None, None)
        if cur_subgraph_a is not None:
            type_start_a = _get_node_target_type(cur_subgraph_a.start_node, gm_a)
        if cur_subgraph_b is not None:
            type_start_b = _get_node_target_type(cur_subgraph_b.start_node, gm_b)
        if cur_subgraph_a is not None and cur_subgraph_b is not None:
            subgraph_relationship = _get_subgraph_relationship_type(cur_subgraph_a, cur_subgraph_b, gm_a, gm_b, type_a_related_to_b)
            if subgraph_relationship == SubgraphTypeRelationship.NOT_RELATED:
                msg = f'\nThe subgraphs\n({cur_subgraph_a}, {type_start_a}) and\n({cur_subgraph_b}, {type_start_b})\nare not related. Please ensure that the two models you pass in have the same number\nof subgraphs, and each pair of subgraphs is related to each other.'
                raise GraphMatchingException(msg)
            elif subgraph_relationship == SubgraphTypeRelationship.EQUAL_BUT_UKNOWN:
                continue
            key_name_a = _get_name_for_subgraph(cur_subgraph_a, gm_a, base_name_to_sets_of_related_ops, existing_names_a)
            key_name_b = _get_name_for_subgraph(cur_subgraph_b, gm_b, base_name_to_sets_of_related_ops, existing_names_b)
            assert key_name_a == key_name_b, f'Subgraph names {key_name_a} and {key_name_b} do not match'
            results[key_name_a] = (cur_subgraph_a, cur_subgraph_b)
            continue
        elif cur_subgraph_a is None and cur_subgraph_b is None:
            break
        else:
            msg = f'\nAttempting to match\n({cur_subgraph_a}, {type_start_a}) and\n({cur_subgraph_b}, {type_start_b}),\none of which is empty. Please ensure that the two models you pass in have the same number\nof subgraphs.'
            raise GraphMatchingException(msg)
    results = collections.OrderedDict(reversed(list(results.items())))
    return results