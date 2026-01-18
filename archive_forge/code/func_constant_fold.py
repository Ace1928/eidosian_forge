import collections
from typing import Any, Callable, Dict, Optional
import torch
import torch.utils._pytree as pytree
@torch.utils._python_dispatch._disable_current_modes()
def constant_fold(gm, constraint_fn: Optional[Callable[[torch.fx.Node], bool]]=None):
    cf = ConstantFolder(gm, skip_constructors=True)
    cf.run()
    for node, constant in cf.node_replacements.items():
        if constraint_fn is not None and (not constraint_fn(node)):
            continue
        replace_node_with_constant(gm, node, constant)
    erased_params = []
    for node in gm.graph.nodes:
        if node.op == 'get_attr' and len(node.users) == 0:
            if hasattr(gm, node.target):
                delattr(gm, node.target)
            erased_params.append(node)
    for node in erased_params:
        gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()