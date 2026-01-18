import copy
from queue import SimpleQueue
from typing import List, Dict, Tuple
import torch.fx
from torch.fx.graph_module import GraphModule
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx.passes.tools_common import NodeList, NodeSet, legalize_graph
from torch.fx.passes.utils import lift_subgraph_as_module
from torch.fx._compatibility import compatibility
def remap_inputs(x):
    if x.op == 'get_attr':
        pass
    if x in nodes:
        return node_map[x]
    if x not in node_to_placeholder:
        placeholder_node = subgraph.placeholder(x.name, type_expr=x.type)
        placeholder_node.meta = copy.copy(x.meta)
        node_to_placeholder[x] = placeholder_node
    return node_to_placeholder[x]