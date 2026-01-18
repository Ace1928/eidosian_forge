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
def remove_no_ops(gm: torch.fx.GraphModule, zeros: Set[torch.fx.Node], ones: Set[torch.fx.Node]):
    """Removes no-ops: (+ 0, - 0, * 1, / 1)"""
    aten = torch.ops.aten
    graph = gm.graph

    def fake_tensors_eq(t1, t2, fields=('shape', 'dtype', 'device')):
        if any((not isinstance(t, torch.Tensor) for t in (t1, t2))):
            return False
        for field in fields:
            if getattr(t1, field) != getattr(t2, field):
                return False
        return True

    def replace_no_op(node, replace_input_index):
        replacement = node.args[replace_input_index]
        if not all((isinstance(arg, torch.fx.Node) for arg in node.args)):
            return
        if not fake_tensors_eq(node.meta['val'], replacement.meta['val']):
            if fake_tensors_eq(node.meta['val'], replacement.meta['val'], ('shape', 'device')):
                with graph.inserting_after(node):
                    replacement = graph.call_function(torch.ops.prims.convert_element_type.default, args=(replacement, node.meta['val'].dtype))
            else:
                return
        node.replace_all_uses_with(replacement)
        replacement.meta.update(node.meta)
        graph.erase_node(node)
    for node in graph.nodes:
        if node.op != 'call_function':
            continue
        if node.target == aten.add.Tensor and len(node.args) == 2:
            if not any((e in zeros for e in node.args)) or node.kwargs.get('alpha', 1) != 1:
                continue
            replace_index = 1 if node.args[0] in zeros else 0
            replace_no_op(node, replace_index)
        elif node.target == aten.sub.Tensor and len(node.args) == 2:
            if node.args[1] not in zeros or node.kwargs.get('alpha', 1) != 1:
                continue
            replace_no_op(node, 0)
        elif node.target == aten.mul.Tensor and len(node.args) == 2:
            if not any((e in ones for e in node.args)):
                continue
            replace_input_index = 1 if node.args[0] in ones else 0
            replace_no_op(node, replace_input_index)
        elif node.target == aten.div.Tensor and len(node.args) == 2 and (node.args[1] in ones):
            replace_no_op(node, 0)