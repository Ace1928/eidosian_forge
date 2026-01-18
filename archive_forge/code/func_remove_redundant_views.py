import logging
import typing
from collections import Counter
from typing import Dict, Set
import torch
import torch._guards
from torch._inductor.constant_folding import ConstantFolder
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from ..pattern_matcher import (
from .replace_random import replace_random_passes
@torch.utils._python_dispatch._disable_current_modes()
def remove_redundant_views(gm: torch.fx.GraphModule):
    """
    Removes redundant views by reusing existing ones.
    """
    views: Dict[torch.fx.Node, Dict[torch.dtype, torch.fx.Node]] = {}
    graph = gm.graph
    for node in graph.nodes:
        if node.op != 'call_function':
            continue
        if node.target != torch.ops.aten.view.dtype:
            continue
        src = node.args[0]
        to_type = node.args[1]
        existing_views = views.get(src)
        is_needed = True
        if existing_views:
            alias = existing_views.get(to_type)
            if alias:
                is_needed = False
                node.replace_all_uses_with(alias)
                alias.meta.update(node.meta)
                graph.erase_node(node)
        else:
            from_type = src.meta['val'].dtype
            existing_views = {from_type: src}
            views[src] = existing_views
        if is_needed:
            existing_views.setdefault(to_type, node)
            views[node] = existing_views
    while True:
        unused_views = []
        for alias in views:
            if not alias.users:
                unused_views.append(alias)
        if len(unused_views) == 0:
            break
        for unused in unused_views:
            views.pop(unused)
            graph.erase_node(unused)