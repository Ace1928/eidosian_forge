from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def node_summary(snode):
    detail = ''
    if isinstance(snode.node, ir.ExternKernelOut):
        detail = f' ({snode.node.kernel})'
    out_tensor_info = ''
    if hasattr(snode.node, 'layout') and hasattr(snode.node.layout, 'size') and hasattr(snode.node.layout, 'stride'):
        out_tensor_info = f' (size={snode.node.layout.size}, stride={snode.node.layout.stride})'
    node_name = ''
    if hasattr(snode.node, 'name'):
        node_name = snode.node.name
    return f'{snode.node.__class__.__name__}{detail}{out_tensor_info} ({node_name})'