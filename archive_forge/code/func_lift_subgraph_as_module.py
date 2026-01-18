from typing import Dict, Tuple
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.nn import Module
@compatibility(is_backward_compatible=False)
def lift_subgraph_as_module(gm: GraphModule, subgraph: Graph, comp_name: str='', class_name: str='GraphModule') -> Tuple[GraphModule, Dict[str, str]]:
    """
    Create a GraphModule for subgraph, which copies the necessary attributes from the original parent graph_module.

    Args:
        gm (GraphModule): parent graph module

        subgraph (Graph): a valid subgraph that contains copied nodes from the parent graph

        comp_name (str): name for the new component

        class_name (str): name for the submodule

    """
    submodule = HolderModule({})
    orig_to_split_fqn_mapping: Dict[str, str] = {}
    for n in subgraph.nodes:
        if n.op not in ('call_module', 'get_attr'):
            continue
        target = n.target
        assert isinstance(target, str)
        target_name_parts = target.split('.')
        curr = submodule
        orig_gm = gm
        for name in target_name_parts[:-1]:
            if not hasattr(curr, name):
                curr.add_module(name, HolderModule({}))
            curr = getattr(curr, name)
            orig_gm = getattr(orig_gm, name)
        leaf_node_name = target_name_parts[-1]
        leaf_node = getattr(orig_gm, leaf_node_name)
        orig_to_split_fqn_mapping[target] = f'{comp_name}.{target}'
        setattr(curr, leaf_node_name, leaf_node)
    return (GraphModule(submodule, subgraph, class_name), orig_to_split_fqn_mapping)