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
@compatibility(is_backward_compatible=False)
def erase_nodes(gm: GraphModule, nodes: NodeList):
    for node in reversed(nodes):
        gm.graph.erase_node(node)