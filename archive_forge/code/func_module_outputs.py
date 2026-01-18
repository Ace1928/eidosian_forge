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
def module_outputs(self) -> Sequence[torch.fx.Node]:
    """Extract module outputs from the sequence of fx nodes this instance holds.

        All nodes that are used by nodes outside of the module are considered module
        outputs. The order of returned module outputs is the same as the their creation order.

        ### Known limitations

        The original ordering of module outputs is not preserved. There is no meta information
        to be found from the `fx.GraphModule` that can be used to recover the original ordering.

        Returns:
            Sequence of module outputs.
        """
    nodes = list(self.fx_nodes())
    assert len(nodes) > 0, 'Cannot extract module inputs from empty nodes.'
    module_outputs: Dict[torch.fx.Node, None] = {}
    node_set: Set[torch.fx.Node] = set(nodes)
    for node in nodes:
        if any((user not in node_set for user in node.users)):
            module_outputs[node] = None
    return list(module_outputs.keys())