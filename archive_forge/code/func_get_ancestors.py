from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def get_ancestors(node):
    ancestors = set()
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node.inverse_users:
                if inp not in ancestors:
                    ancestors.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return ancestors