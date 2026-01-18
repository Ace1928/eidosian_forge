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
def fuse_by_partitions(gm: GraphModule, partitions: List[NodeList]) -> GraphModule:
    for partition_id, nodes in enumerate(partitions):
        sorted_nodes = topo_sort(nodes)
        submodule_name = 'fused_' + str(partition_id)
        sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(gm, sorted_nodes, submodule_name)
        insert_subgm(gm, sub_gm, orig_inputs, orig_outputs)
        erase_nodes(gm, sorted_nodes)
    legalize_graph(gm)
    return gm