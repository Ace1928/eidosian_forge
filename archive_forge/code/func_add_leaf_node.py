from __future__ import annotations
import abc
import collections
import copy
import operator
from typing import (
import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree
def add_leaf_node(self, leaf_node: _LeafNode) -> None:
    """Adds a leaf node to the module.

        The leaf node must belong to the same or a child module. This method will recursively
        construct _ModuleNode instance based on the stack_meta information of the leaf node.
        """
    if self.is_same_module_as(leaf_node) or leaf_node.fx_op == 'call_module':
        self._nodes.append(leaf_node)
    elif self.is_parent_module_of(leaf_node):
        last_node = self._nodes[-1] if self._nodes else None
        if isinstance(last_node, _ModuleNode) and (last_node.is_parent_module_of(leaf_node) or last_node.is_same_module_as(leaf_node)):
            last_node.add_leaf_node(leaf_node)
        else:
            stack_meta = copy.deepcopy(self.stack_meta)
            stack_meta.push(leaf_node.stack_meta[len(self.stack_meta)])
            last_node = _ModuleNode(self._reference_module, stack_meta)
            self._nodes.append(last_node)
            last_node.add_leaf_node(leaf_node)
    else:
        raise AssertionError(f'Node {leaf_node} ({leaf_node.stack_meta}) does not belong to module {self._stack_meta}.')