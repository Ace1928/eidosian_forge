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
def run_from(self, node_idx):
    module_idx = 0
    while node_idx < len(self.nodes):
        node = self.nodes[node_idx]
        assert node.op != 'placeholder'
        self.print()
        self.print('STEP', node_idx, node.format_node())
        self.print(self.module_stack)
        if node.op == 'output':
            if len(self.module_stack) == 1:
                return node_idx
            self.finalize_outputs()
            return node_idx
        node_module_stack = [path for path, ty in node.meta['nn_module_stack'].values()] if 'nn_module_stack' in node.meta else self.module_stack
        if node_module_stack[:len(self.module_stack)] != self.module_stack:
            self.finalize_outputs()
            self.print('outlining', self.fqn)
            self.print(self.graph)
            return node_idx
        assert node_module_stack is not None
        if is_prefix(self.module_stack, node_module_stack):
            next_module = node_module_stack[len(self.module_stack)]
            self.print('Creating new stack frame for', next_module)
            node_idx = ModuleFrame(self.flat_graph, self.seen_nodes, self.seen_modules, self, self.module_stack + [next_module], list(node.meta['nn_module_stack'].keys())[len(self.module_stack)], self.module_call_graph).run_from(node_idx)
            module_idx += 1
            continue
        assert node_module_stack == self.module_stack
        self.copy_node(node)
        node_idx += 1