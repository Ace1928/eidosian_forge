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
def remap_input(self, x):
    assert x.graph is self.flat_graph
    if x in self.node_map:
        return self.node_map[x]
    if x not in self.node_to_placeholder:
        self.add_placeholder(x)
        if self.parent_call_module is not None:
            self.parent_call_module.insert_arg(0, self.parent.remap_input(x))
    return self.node_to_placeholder[x]