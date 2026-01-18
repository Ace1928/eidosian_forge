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
def insert_subgm(gm: GraphModule, sub_gm: GraphModule, orig_inputs: Tuple[Node, ...], orig_outputs: Tuple[Node, ...]):
    submodule_name = sub_gm.__class__.__name__
    gm.add_submodule(submodule_name, sub_gm)
    module_node = gm.graph.call_module(submodule_name, args=orig_inputs, kwargs=None)
    if len(orig_outputs) == 1:
        orig_outputs[0].replace_all_uses_with(module_node, propagate_meta=True)
    else:
        for i, orig_output in enumerate(orig_outputs):
            proxy_out = torch.fx.Proxy(module_node)[i].node
            orig_output.replace_all_uses_with(proxy_out, propagate_meta=True)
    return gm