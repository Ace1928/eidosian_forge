import re
from typing import Callable, Dict, Optional, Set, Union
import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module
def split_const_subgraphs(module: Union[torch.nn.Module, torch.fx.GraphModule], skip_folding_node_fn: Optional[Callable[[torch.fx.Node], bool]]=None, device_for_folded_attrs: str='cpu') -> FoldedGraphModule:
    """
    Looks through `module` for any nodes that have all constant attribute inputs
    and separates them out into their own constant subgraph, and returns a
    FoldedGraphModule which runs that constant subgraph on the first run to set
    attributes on the module prior to running the non-constant portion of the
    graph.
    """
    if not isinstance(module, torch.fx.GraphModule):
        mod_traced = torch.fx.symbolic_trace(module)
    else:
        mod_traced = module
    const_nodes: Set[torch.fx.Node] = set()
    found_const_folding = False
    for node in mod_traced.graph.nodes:
        if node.op in {'placeholder', 'output'}:
            continue
        if node.op != 'get_attr' and (not set(node.all_input_nodes).issubset(const_nodes)):
            continue
        if skip_folding_node_fn and skip_folding_node_fn(node):
            continue
        if node.is_impure():
            continue
        const_nodes.add(node)
        if node.op != 'get_attr':
            found_const_folding = True
    if not found_const_folding:
        return FoldedGraphModule(mod_traced, mod_traced.graph)

    def mod_partition(node: torch.fx.Node):
        return 0 if node in const_nodes else 1
    split = split_module(mod_traced, module, mod_partition)
    const_gm, non_const_gm = (split.submod_0, split.submod_1)
    const_mod_name, non_const_mod_name = ('submod_0', 'submod_1')
    for node in non_const_gm.graph.nodes:
        if node.op == 'call_module':
            setattr(split, node.target, getattr(non_const_gm, node.target))
    for node in const_gm.graph.nodes:
        if node.op == 'call_module':
            setattr(split, node.target, getattr(const_gm, node.target))
    call_const_gm_args = None
    for node in split.graph.nodes:
        if node.op == 'call_module':
            if node.target == const_mod_name:
                call_const_gm_args = node.args
                break
    assert call_const_gm_args is not None
    root_const_gm = torch.fx.GraphModule(split, const_gm.graph)
    for node in root_const_gm.graph.nodes:
        if node.op == 'output':
            multiple_outputs = isinstance(node.args[0], tuple)
            continue
        if node.op != 'placeholder':
            continue
        in_node = next((n for n in call_const_gm_args if n.name == node.target))
        assert in_node.op == 'get_attr'
        with root_const_gm.graph.inserting_before(node):
            new_node = root_const_gm.graph.get_attr(in_node.target)
        new_node.meta = node.meta.copy()
        node.replace_all_uses_with(new_node)
        root_const_gm.graph.erase_node(node)
    assert 'multiple_outputs' in locals()
    fx_const_folded_attrs_name = get_unique_attr_name_in_module(split, '_FX_CONST_FOLDED_ATTRS')
    setattr(split, fx_const_folded_attrs_name, torch.nn.ParameterList() if multiple_outputs else torch.nn.Parameter())
    for node in split.graph.nodes:
        if node.op == 'call_module' and node.target == const_mod_name:
            with node.graph.inserting_before(node):
                folded_attrs = node.graph.get_attr(fx_const_folded_attrs_name)
            folded_attrs.meta = node.meta.copy()
            node.replace_all_uses_with(folded_attrs)
            break
    split.graph.eliminate_dead_code()
    _inline_module(split, non_const_mod_name)
    return FoldedGraphModule(split, split.graph, root_const_gm.graph, fx_const_folded_attrs_name, device_for_folded_attrs)