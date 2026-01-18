import inspect
import logging
import torch
from torch._ops import HigherOrderOperator
from torch.utils.checkpoint import checkpoint, uid
import torch._dynamo.config
def tag_nodes(self, gmod):
    unique_graph_id = next(uid)
    for node in gmod.graph.nodes:
        if node.op in ('call_function', 'call_method', 'call_module'):
            node.meta['recompute'] = unique_graph_id
    return gmod