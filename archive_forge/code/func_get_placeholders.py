import torch
import torch.fx as fx
from torch.utils._pytree import tree_flatten
from torch.utils import _pytree as pytree
def get_placeholders(graph):
    return list(filter(lambda x: x.op == 'placeholder', graph.nodes))