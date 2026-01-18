import copy
import operator
from copy import deepcopy
from typing import cast, Dict, List, Optional, Union
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.exported_program import (
from torch.fx import GraphModule
from .utils import _check_input_constraints_pre_hook
def graph_dump(graph: torch.fx.Graph) -> str:
    ret = []
    nodes_idx: Dict[int, int] = {}

    def arg_dump(arg) -> str:
        if isinstance(arg, torch.fx.Node):
            return '%' + str(nodes_idx[id(arg)])
        return str(arg)
    for i, node in enumerate(graph.nodes):
        args_dump = [str(arg) for arg in pytree.tree_map(arg_dump, node.args)]
        args_dump += [f'{key}={value}' for key, value in pytree.tree_map(arg_dump, node.kwargs).items()]
        target = node.target if node.op == 'call_function' else ''
        ret.append(f'{i}: {node.op}[{target}]({', '.join(args_dump)})')
        nodes_idx[id(node)] = i
    return '\n'.join(ret)