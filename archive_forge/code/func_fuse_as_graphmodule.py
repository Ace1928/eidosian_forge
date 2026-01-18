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
def fuse_as_graphmodule(gm: GraphModule, nodes: NodeList, module_name: str) -> Tuple[GraphModule, Tuple[Node, ...], Tuple[Node, ...]]:
    """
    Fuse nodes in graph_module into a GraphModule.

    Args:
        gm (GraphModule): target graph_module

        nodes (List[Node]): list of nodes in `gm` to fuse, where the node must be topologically sorted

        module_name: class name for the fused GraphModule

    Returns:
        fused_gm (GraphModule): fused graph module, where its node is a copy of `nodes` in `gm`

        original_inputs (Tuple[Node, ...]): input nodes to `nodes` in original `gm`

        original_outputs (Tuple[Node, ...]): consumer nodes of `nodes` in original `gm`

    """
    for node in nodes:
        assert node.graph.owning_module is gm, f"{node} doesn't belong to passed in graph module {gm._get_name()}"
        assert not node._erased, f'{node} has been removed from owning graph'
        assert node in gm.graph.nodes, f'{node} is not found in graph module {gm._get_name()}'
    assert validate_partition(nodes), 'Invalid partition, found dependency cycles'
    subgraph = Graph()
    node_to_placeholder: Dict[Node, Node] = {}
    node_map: Dict[Node, Node] = {}

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
    for node in nodes:
        new_node = subgraph.node_copy(node, remap_inputs)
        node_map[node] = new_node
    output_mapping: Dict[Node, Node] = {}
    for node in nodes:
        for user_node in node.users:
            if user_node not in nodes:
                output_mapping[node] = node_map[node]
    outs = tuple(output_mapping.values())
    subgraph.output(outs[0] if len(outs) == 1 else outs)
    subgraph.lint()
    fused_gm: GraphModule
    fused_gm, _ = lift_subgraph_as_module(gm, subgraph, comp_name='', class_name=module_name)
    original_inputs: Tuple[Node, ...] = tuple(node_to_placeholder.keys())
    original_outputs: Tuple[Node, ...] = tuple(output_mapping.keys())
    return (fused_gm, original_inputs, original_outputs)