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
def copy_node(self, node):
    self.print('copying', node.format_node())
    self.node_map[node] = self.graph.node_copy(node, self.remap_input)
    self.seen_nodes[node.name] = node