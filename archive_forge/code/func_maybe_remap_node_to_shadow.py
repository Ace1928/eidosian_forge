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
def maybe_remap_node_to_shadow(node):
    """
            If unshadowed `node` has a shadow version, return that. If not,
            return `node`.
            """
    if not isinstance(node, Node):
        return node
    if node.op in ('placeholder', 'get_attr'):
        return node
    prev_subgraph = _get_subgraph_containing_node(node, subgraphs_dedup)
    if prev_subgraph is None:
        prev_subgraph = [node]
    prev_first_node = prev_subgraph[0]
    prev_shadow_output = orig_first_node_to_shadow_out_node[prev_first_node]
    return prev_shadow_output